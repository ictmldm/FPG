import torch

#device = torch.device("cuda:1")

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.

    Args:
        tokens_a (list): List of tokens for the first sequence.
        tokens_b (list): List of tokens for the second sequence.
        max_length (int): The maximum total length for the sequence pair.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,
                 extraction_mask=None, extraction_start_ids=None, extraction_end_ids=None,
                 augmentation_mask=None, augmentation_start_ids=None, augmentation_end_ids=None):
        """
        Initializes an InputFeatures object.

        Args:
            input_ids (list): List of token IDs.
            input_mask (list): List of attention mask values (1 for real tokens, 0 for padding).
            segment_ids (list): List of segment IDs (0 for first sequence, 1 for second).
            extraction_mask (list, optional): Mask for extraction span.
            extraction_start_ids (list, optional): Start indices for extraction.
            extraction_end_ids (list, optional): End indices for extraction.
            augmentation_mask (list, optional): Mask for augmentation span.
            augmentation_start_ids (list, optional): Start indices for augmentation.
            augmentation_end_ids (list, optional): End indices for augmentation.
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_mask = augmentation_mask
        self.augmentation_start_ids = augmentation_start_ids
        self.augmentation_end_ids = augmentation_end_ids

def convert_examples_to_features(claims, docs, max_seq_length,
                                 tokenizer,
                                 device,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True
                                 ):
    """
    Loads a data file into a list of `InputBatch`s.

    Args:
        claims (list): List of claim strings.
        docs (list): List of document strings.
        max_seq_length (int): The maximum sequence length.
        tokenizer: The tokenizer to use.
        device (torch.device): The device to use for tensors.
        cls_token_at_end (bool, optional): Whether the CLS token is at the end. Defaults to False.
        cls_token (str, optional): The CLS token string. Defaults to '[CLS]'.
        cls_token_segment_id (int, optional): The segment ID for the CLS token. Defaults to 1.
        sep_token (str, optional): The SEP token string. Defaults to '[SEP]'.
        sep_token_extra (bool, optional): Whether to add an extra SEP token. Defaults to False.
        pad_on_left (bool, optional): Whether to pad on the left. Defaults to False.
        pad_token (int, optional): The ID of the padding token. Defaults to 0.
        pad_token_segment_id (int, optional): The segment ID for the padding token. Defaults to 0.
        sequence_a_segment_id (int, optional): The segment ID for sequence A. Defaults to 0.
        sequence_b_segment_id (int, optional): The segment ID for sequence B. Defaults to 1.
        mask_padding_with_zero (bool, optional): Whether to mask padding with zero. Defaults to True.

    Returns:
        dict: A dictionary containing input tensors ('input_ids', 'attention_mask', 'token_type_ids').
    """
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for i in range(len(claims)):
        # tokens_a 为 context, tokens_b 为 claim
        # tokens_a is context, tokens_b is claim
        tokens_a = tokenizer.tokenize(docs[i])
        tokens_b = tokenizer.tokenize(claims[i])
        
        special_tokens_count = 4 if sep_token_extra else 3
        # 如果序列过长，就进行裁剪
        # Truncate sequences if they are too long
        if len(tokens_a) + len(tokens_b)> max_seq_length - special_tokens_count:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        #extraction_span_len = len(tokens_a) + 2
        #extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]
        #augmentation_mask = [1 if extraction_span_len <= ix < extraction_span_len + len(tokens_b) + 1  else 0 for ix in range(max_seq_length)]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        feature = InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          #extraction_mask=extraction_mask,
                          #augmentation_mask=augmentation_mask
                          )
        features.append(feature)
    
    inputs = dict()
    inputs['input_ids'] = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda(device)
    inputs['attention_mask'] = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda(device)
    inputs['token_type_ids'] = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda(device)
    #inputs["ext_mask"] = torch.tensor([f.extraction_mask for f in features], dtype=torch.float).cuda(device)
    #inputs["aug_mask"] = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float).cuda(device)

    return inputs