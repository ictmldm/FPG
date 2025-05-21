import torch
import numpy as np
from config.run_config import device


class Beam:
    """ Beam search class for managing beams during decoding """

    def __init__(self, size, pad, eos, device=device):
        """
        Initializes a Beam object.

        Args:
            size (int): The size of the beam.
            pad (int): The ID of the padding token.
            eos (int): The ID of the end-of-sequence token.
            device (torch.device): The device to use for tensors.
        """
        self.size = size  # Beam size
        self._done = False  # Flag to indicate if the beam search is complete
        self.PAD = pad  # Padding token ID
        self.EOS = eos  # End-of-sequence token ID
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)  # Scores for each beam
        self.all_scores = []  # List to store all scores
        self.prev_ks = []  # Backpointers at each time-step
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]  # Outputs at each time-step
        self.next_ys[0][0] = self.EOS  # Initialize with [EOS, PAD, PAD, ...]

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        """Check if beam search is complete."""
        return self._done

    def advance(self, word_logprob):    # beam * vocab
        """
        Update beam status and check if finished or not.

        Args:
            word_logprob (torch.Tensor): Log probabilities of the next word for each beam.
                                         Shape: (beam_size, vocab_size)

        Returns:
            bool: True if the beam search is complete for this instance, False otherwise.
        """
        num_words = word_logprob.size(1)  # Vocabulary size

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # in initial case,
            beam_lk = word_logprob[0]

        # Flatten the beam log probabilities
        flat_beam_lk = beam_lk.view(-1)     # vocab
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)   # size, size

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # Calculate the beam and word indices from the flat indices
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # Check if the top beam ends with EOS
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.EOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis by walking back through the backpointers.

        Args:
            k (int): The index of the hypothesis in the current step.

        Returns:
            list: The list of token IDs representing the hypothesis.
        """
        # print(k.type())
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, h_src, h_mask, src, src_mask, max_len, pad, eos, beam_size, device):
    """
    Perform beam search for sequence generation.

    Args:
        model (torch.nn.Module): The model to use for decoding.
        h_src (torch.Tensor): Encoded history source tensor.
        h_mask (torch.Tensor): Attention mask for history source.
        src (torch.Tensor): Encoded source tensor.
        src_mask (torch.Tensor): Attention mask for source.
        max_len (int): Maximum length of the generated sequence.
        pad (int): ID of the padding token.
        eos (int): ID of the end-of-sequence token.
        beam_size (int): The size of the beam.
        device (torch.device): The device to use for tensors.

    Returns:
        tuple: A tuple containing:
            - list: List of generated hypotheses (token IDs).
            - list: List of scores for the generated hypotheses.
    """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ Map instance indices to tensor positions. """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """
        Collect tensor parts associated with active instances.

        Args:
            beamed_tensor (torch.Tensor): The tensor to collect parts from.
            curr_active_inst_idx (list): List of indices of currently active instances.
            n_prev_active_inst (int): Number of previously active instances.
            n_bm (int): Beam size.

        Returns:
            torch.Tensor: Tensor containing parts for active instances.
        """
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # active instances (elements of batch) * beam search size x seq_len x h_dimension
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # select only parts of tensor which are still active
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_enc, user, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        """
        Collect information for active instances.

        Args:
            src_enc (torch.Tensor): Encoded source tensor.
            user (torch.Tensor, optional): User vector tensor.
            src_mask (torch.Tensor): Attention mask for source.
            inst_idx_to_position_map (dict): Mapping from instance index to tensor position.
            active_inst_idx_list (list): List of indices of currently active instances.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Active encoded source tensor.
                - torch.Tensor: Active user vector tensor (or None).
                - torch.Tensor: Active source attention mask.
                - dict: Updated mapping from instance index to tensor position.
        """
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)
        if user != None:
            active_user = collect_active_part(user, active_inst_idx, n_prev_active_inst, beam_size)
        else:
            active_user = None

        return active_src_enc, active_user, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, user, src_mask, inst_idx_to_position_map, n_bm):
        """
        Decode and update beam status, and then return active beam idx.

        Args:
            inst_dec_beams (list): List of Beam objects.
            len_dec_seq (int): Current length of the decoded sequence.
            enc_output (torch.Tensor): Encoder output tensor.
            user (torch.Tensor, optional): User vector tensor.
            src_mask (torch.Tensor): Attention mask for source.
            inst_idx_to_position_map (dict): Mapping from instance index to tensor position.
            n_bm (int): Beam size.

        Returns:
            list: List of indices of active instances.
        """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            """ Prepare the partial sequence for each beam. """
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # Batch size x Beam size x Dec Seq Len
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # Batch size*Beam size x Dec Seq Len
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, user, src_mask, n_active_inst, n_bm):
            """ Predict the next word for each beam. """
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            bart_output = model.get_decoder()(
                input_ids=dec_seq,
                encoder_hidden_states=enc_output,
                user_vec=user,
                encoder_attention_mask=src_mask
            )
            lm_logits = model.lm_head(bart_output[0]) + model.final_logits_bias
            word_logprob = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            # word_logprob = torch.nn.functional.softmax(lm_logits, dim=-1)
            # word_logprob = model.generator(word_logprob[:, -1])
            word_logprob = word_logprob[:, -1, :].view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            """ Collect the indices of active instances. """
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])  # Fill Beam object with assigned probabilities
                if not is_inst_complete:  # if top beam ended with eos, we do not add it
                    active_inst_idx_list.append(inst_idx)

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # get decoding sequence for each beam
        # size: Batch size*Beam size x Dec Seq Len
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # get word probabilities for each beam
        # size: Batch size x Beam size x Vocabulary
        word_logprob = predict_word(dec_seq, enc_output, user, src_mask, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        """
        Collect the final hypotheses and scores.

        Args:
            inst_dec_beams (list): List of Beam objects.
            n_best (int): Number of best hypotheses to collect.

        Returns:
            tuple: A tuple containing:
                - list: List of generated hypotheses (token IDs).
                - list: List of scores for the generated hypotheses.
        """
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores.append(scores[:n_best])

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp.append(hyps)
        return all_hyp, all_scores

    with torch.no_grad():
        # Encode the input sequences
        encoder_outputs, user = model.get_encoder()(
            h_inputs_ids=h_src,
            h_attention_mask=h_mask,
            input_ids=src,
            attention_mask=src_mask
        )
        src_enc = encoder_outputs[0]

        # Repeat data for beam search
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat_interleave(beam_size, dim=0)
        user = user.repeat_interleave(beam_size, dim=0) if user is not None else None
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)

        # Initialize beams
        inst_dec_beams = [Beam(beam_size, pad, eos, device) for _ in range(batch_size)]

        # Initialize active instance bookkeeping
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # Start decoding
        for len_dec_seq in range(1, max_len + 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, user, src_mask, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            # filter out inactive tensor parts (for already decoded sequences)
            src_enc, user, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, user, src_mask, inst_idx_to_position_map, active_inst_idx_list)


    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)
    return batch_hyp, batch_scores
