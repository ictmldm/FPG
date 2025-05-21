import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartTokenizer
from config.run_config import (
    processed_data_path, news_file_path_sp, train_file_path, dev_file_path,
    test_file_path, bart_name, max_click_length, max_news_title_length,
    max_news_body_length, limit, news_file_path
)
import os
import logging
from datetime import datetime

# Configure logging
log_filename = f"./logs/data_preprocessing.log"
# _{datetime.now().strftime('%Y%m%d_%H%M%S')}
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check and create directory for processed data
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)
    logging.info(f"Created directory: {processed_data_path}")


def load_data(file_path, data_type='news'):
    """
    Loads data from a tab-separated file.

    Args:
        file_path (str): Path to the data file.
        data_type (str): Type of data being loaded (e.g., 'news', 'train').

    Returns:
        pd.DataFrame: Loaded data.
    """
    logging.info(f"Loading {data_type} data: {file_path}")
    data = pd.read_csv(file_path, sep='\t')
    data.fillna(value=" ", inplace=True)
    logging.info(f"{data_type} data loaded successfully")
    return data


def prepare_tokenizer(bart_name):
    """
    Prepares the Bart tokenizer.

    Args:
        bart_name (str): Name or path of the pre-trained Bart model.

    Returns:
        BartTokenizer: Loaded Bart tokenizer.
    """
    logging.info(f"Loading Bart tokenizer: {bart_name}")
    tokenizer = BartTokenizer.from_pretrained(
        bart_name,
        trust_remote_code=True,
        local_files_only=True,
        do_lower_case=True
    )
    logging.info("Bart tokenizer loaded successfully")
    return tokenizer


def build_news_dict(news_ids):
    """
    Builds a dictionary mapping news IDs to indices.

    Args:
        news_ids (list): List of news IDs.

    Returns:
        dict: Dictionary mapping news ID to index (1-based).
    """
    logging.info("Building news index dictionary")
    news_dict = {news_id: idx + 1 for idx, news_id in enumerate(news_ids)}
    logging.info("News index dictionary built successfully")
    return news_dict


def process_test_data(test, news_titles, news_bodys, news_dict, tokenizer,
                     max_click_length, max_news_title_length, max_news_body_length):
    """
    Processes test data, encoding click history, news bodies, and collecting positive titles.

    Args:
        test (pd.DataFrame): Test data.
        news_titles (np.ndarray): Array of news titles.
        news_bodys (np.ndarray): Array of news bodies.
        news_dict (dict): Dictionary mapping news IDs to indices.
        tokenizer (BartTokenizer): Bart tokenizer.
        max_click_length (int): Maximum length of click history.
        max_news_title_length (int): Maximum length for news titles.
        max_news_body_length (int): Maximum length for news bodies.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Processed test samples.
            - set: Set of unique news IDs used in the test set.
    """
    logging.info("Starting test data processing")
    # process testset
    h_inputs = []
    h_masks = []
    bodys = []
    bodys_masks = []
    p_titles = []
    test_ids = set()  # Use a set to store unique test IDs
    pbar = tqdm(range(len(test)), desc="Processing Test Data")
    # user_id, click_history_ids, pos_news_ids, per_titles
    for i in pbar:
        _, click_history_ids, pos_ids, titles = test.iloc[i]
        # convert ids to titles
        click_history_ids = click_history_ids.split(",")
        click_history = [news_titles[news_dict[id] - 1] for id in click_history_ids]
        # pad or cut click history to max_click_len
        if len(click_history) < max_click_length:
            padding_length = max_click_length - len(click_history)
            h_mask = [0] * padding_length + [1] * len(click_history)
            click_history = [tokenizer.pad_token] * padding_length + click_history
        else:
            click_history = click_history[-max_click_length:]
            h_mask = [1] * max_click_length

        # encode click history
        click_history_encoded = tokenizer.batch_encode_plus(
            click_history,
            max_length=max_news_title_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        history_input_ids = click_history_encoded['input_ids']
        history_mask = torch.tensor(h_mask)

        # news body & title
        pos_ids = pos_ids.split(",")
        titles = titles.split(";;")
        assert len(pos_ids) == len(titles)
        for pos_id, title in zip(pos_ids, titles):
            # ignore empty title
            if not title.strip():
                continue
            body = news_bodys[news_dict[pos_id] - 1]
            # ignore empty body
            if not body.strip():
                continue
            test_ids.add(pos_id)  # Add to test_ids set

            body_encoded = tokenizer(
                body,
                max_length=max_news_body_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            body_input_ids = body_encoded['input_ids']
            body_mask = body_encoded['attention_mask']

            h_inputs.append(history_input_ids.tolist())
            h_masks.append(history_mask.tolist())
            bodys.append(body_input_ids.squeeze(0).tolist())
            bodys_masks.append(body_mask.squeeze(0).tolist())
            p_titles.append(title)

    print(f"test_ids: {len(test_ids)}")
    logging.info("Test data processing completed")
    logging.info(f"Total test data IDs: {len(test_ids)}")
    test_samples = pd.DataFrame({
        'h_inputs': h_inputs,
        'h_masks': h_masks,
        'bodys': bodys,
        'bodys_masks': bodys_masks,
        'p_titles': p_titles
    })
    print(f"test_samples: {len(test_samples)}")
    test_samples.to_feather(os.path.join(processed_data_path, 'test.feather'))
    logging.info(f"Test data saved successfully, total {len(test_samples)} samples")

    return test_samples, test_ids  # Return test_ids for exclusion


def process_click_history(click_history_ids, news_titles, news_dict, tokenizer,
                         max_click_length, data_type="train"):
    """
    Processes click history, converting IDs to titles and handling padding/truncation.

    Args:
        click_history_ids (str): Comma or space separated string of click history news IDs.
        news_titles (np.ndarray): Array of news titles.
        news_dict (dict): Dictionary mapping news IDs to indices.
        tokenizer (BartTokenizer): Bart tokenizer.
        max_click_length (int): Maximum length of click history.
        data_type (str): Type of data ('train' or 'test') to handle different separators.

    Returns:
        tuple: A tuple containing:
            - list: Processed click history titles.
            - list: Attention mask for the click history.
    """
    # convert ids to titles
    click_history_ids = click_history_ids.split(",") if data_type == "test" else click_history_ids.split(" ")
    click_history = [news_titles[news_dict[id] - 1] for id in click_history_ids]

    # pad or cut click history to max_click_len
    if len(click_history) < max_click_length:
        padding_length = max_click_length - len(click_history)
        click_history = [tokenizer.pad_token] * padding_length + click_history
        h_mask = [0] * padding_length + [1] * len(click_history_ids)
    else:
        click_history = click_history[-max_click_length:]
        h_mask = [1] * max_click_length

    return click_history, h_mask


def encode_text(text, tokenizer, max_length, replace_first_token=False):
    """
    Encodes text using the tokenizer.

    Args:
        text (str or list): Text or list of texts to encode.
        tokenizer (BartTokenizer): Bart tokenizer.
        max_length (int): Maximum length for encoding.
        replace_first_token (bool): Whether to replace the first token with EOS token ID.

    Returns:
        tuple: A tuple containing:
            - list: Encoded input IDs.
            - list: Attention mask.
    """
    text_encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    if replace_first_token:
        # Replace the first token (usually BOS) with EOS token ID for title generation target
        text_encoded['input_ids'][0][0] = tokenizer.eos_token_id
    return text_encoded['input_ids'].squeeze(0).tolist(), text_encoded['attention_mask'].squeeze(0).tolist()


def process_samples(samples, news_titles, news_bodys, news_dict, tokenizer,
                   max_click_length, max_news_title_length, max_news_body_length, limit=None):
    """
    Processes sample data (train/validation), encoding inputs and targets.

    Args:
        samples (pd.DataFrame): Sample data.
        news_titles (np.ndarray): Array of news titles.
        news_bodys (np.ndarray): Array of news bodies.
        news_dict (dict): Dictionary mapping news IDs to indices.
        tokenizer (BartTokenizer): Bart tokenizer.
        max_click_length (int): Maximum length of click history.
        max_news_title_length (int): Maximum length for news titles.
        max_news_body_length (int): Maximum length for news bodies.
        limit (int, optional): Maximum number of samples per news ID. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - list: Processed history input IDs.
            - list: Processed history masks.
            - list: Processed body input IDs.
            - list: Processed body masks.
            - list: Processed title input IDs.
            - list: Processed title masks.
            - dict: Count of samples per news ID.
            - dict: Count of samples per user ID.
    """
    logging.info("Starting sample data processing")
    h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks = [], [], [], [], [], []
    news_count, user_count = {}, {}
    pbar = tqdm(range(len(samples)), desc="Processing Samples")

    for i in pbar:
        _, click_history_ids, _, _, pos_ids, _, _, _, _ = samples.iloc[i]

        click_history, h_mask = process_click_history(
            click_history_ids, news_titles, news_dict, tokenizer,
            max_click_length
        )
        pos_ids = pos_ids.split(" ")
        history_input_ids, _ = encode_text(
            click_history, tokenizer, max_news_title_length
        )

        for pos_id in pos_ids:
            body = news_bodys[news_dict[pos_id] - 1]
            title = news_titles[news_dict[pos_id] - 1]
            # ignore empty title and body
            if len(title.strip()) == 0 or len(body.strip()) == 0:
                continue
            if limit and pos_id in news_count and news_count[pos_id] >= limit:
                continue
            if pos_id in news_count:
                news_count[pos_id] += 1
            else:
                news_count[pos_id] = 1
            if i in user_count:
                user_count[i] += 1
            else:
                user_count[i] = 1

            body_input_ids, body_mask = encode_text(body, tokenizer, max_news_body_length)
            title_input_ids, title_mask = encode_text(
                title, tokenizer, max_news_title_length, replace_first_token=True
            )

            h_inputs.append(history_input_ids if limit is not None else click_history)
            h_masks.append(h_mask)
            bodys.append(body_input_ids)
            bodys_masks.append(body_mask)
            titles.append(title_input_ids)
            titles_masks.append(title_mask)

    logging.info("Sample data processing completed")
    return h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, news_count, user_count


def process_raw_test(test, news_titles, news_bodys, news_dict, tokenizer, max_click_length):
    """
    Processes raw test data, collecting original history, bodies, and titles.

    Args:
        test (pd.DataFrame): Raw test data.
        news_titles (np.ndarray): Array of news titles.
        news_bodys (np.ndarray): Array of news bodies.
        news_dict (dict): Dictionary mapping news IDs to indices.
        tokenizer (BartTokenizer): Bart tokenizer.
        max_click_length (int): Maximum length of click history.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Processed raw test samples.
            - set: Set of unique news IDs used in the raw test set.
    """
    logging.info("Starting raw test data processing")
    h_inputs, bodys, o_titles, p_titles, test_ids = [], [], [], [], set()

    for _, click_history_ids, pos_ids, titles in tqdm(test.itertuples(index=False), total=len(test), desc="Processing Raw Test Data"):
        click_history, _ = process_click_history(
            click_history_ids, news_titles, news_dict, tokenizer,
            max_click_length, data_type="test"
        )
        pos_ids, titles = pos_ids.split(","), titles.split(";;")
        assert len(pos_ids) == len(titles)

        for pos_id, title in zip(pos_ids, titles):
            if not title.strip():
                continue
            body = news_bodys[news_dict[pos_id] - 1]
            o_title = news_titles[news_dict[pos_id] - 1]
            if not body.strip():
                continue
            test_ids.add(pos_id)

            h_inputs.append(click_history)
            bodys.append(body)
            o_titles.append(o_title)
            p_titles.append(title)

    test_samples = pd.DataFrame({
        'history': h_inputs,
        'bodys': bodys,
        'o_titles': o_titles,
        'p_titles': p_titles
    })
    test_samples.to_feather(os.path.join(processed_data_path, 'raw_test.feather'))
    logging.info("Raw test data processing completed")
    return test_samples, test_ids


def save_samples_to_feather(file_name, h_inputs, h_masks, bodys, bodys_masks, titles=None, titles_masks=None):
    """
    Saves sample data to a Feather file.

    Args:
        file_name (str): Name of the output feather file.
        h_inputs (list): List of history input IDs.
        h_masks (list): List of history masks.
        bodys (list): List of body input IDs.
        bodys_masks (list): List of body masks.
        titles (list, optional): List of title input IDs. Defaults to None.
        titles_masks (list, optional): List of title masks. Defaults to None.
    """
    logging.info(f"Saving sample data to: {file_name}")
    data = {
        'h_inputs': h_inputs,
        'h_masks': h_masks,
        'bodys': bodys,
        'bodys_masks': bodys_masks
    }
    if titles and titles_masks:
        data['titles'] = titles
        data['titles_masks'] = titles_masks
    samples = pd.DataFrame(data)
    samples.to_feather(os.path.join(processed_data_path, file_name))
    logging.info(f"Sample data saved successfully: {file_name}")


def main():
    logging.info("Starting data loading")
    news = load_data(news_file_path_sp)
    train = load_data(train_file_path, 'train')
    valid = load_data(dev_file_path, 'valid')
    test = load_data(test_file_path, 'test')

    news_ids, news_titles, news_bodys = news["News ID"].values, news["Headline"].values, news["News body"].values
    news_dict = build_news_dict(news_ids)
    tokenizer = prepare_tokenizer(bart_name)

    # Process test data and get news IDs used in test set
    logging.info("Processing test data")
    test_samples, test_ids = process_test_data(
        test, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length
    )

    # Process training data and get news IDs used in training set
    h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, train_news_count, train_user_count = process_samples(
        train, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length, limit=limit
    )
    save_samples_to_feather(f"train_limit_to_{limit}.feather", h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks)
    logging.info(f"Training data processing completed: {len(train)} samples")
    logging.info(f"{len(train_news_count)} news, {len(train_user_count)} users")

    # Process validation data
    logging.info("Processing validation data")
    h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, valid_news_count, valid_user_count = process_samples(
        valid, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length
    )
    save_samples_to_feather("valid.feather", h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks)
    logging.info(f"Validation data processing completed: {len(valid)} samples")
    logging.info(f"{len(valid_news_count)} news, {len(valid_user_count)} users")

    # Collect all news IDs to exclude (from training and test)
    used_news_ids = set(train_news_count.keys()).union(set(test_ids))
    logging.info(f"Pre-training data will exclude {len(used_news_ids)} news IDs from training and test data")

    # Process pre-training data
    logging.info("Processing pre-training data")
    body_inputs, body_masks, title_inputs, title_masks = [], [], [], []
    for id in tqdm(news_ids, desc="Processing Pretrain Data"):
        # if id in used_news_ids:
        #     continue # This line is commented out, so all news are included
        body = news_bodys[news_dict[id] - 1]
        title = news_titles[news_dict[id] - 1]
        if len(title.strip()) == 0 or len(body.strip()) == 0:
            continue
        body_input_ids, body_mask = encode_text(body, tokenizer, max_news_body_length)
        title_input_ids, title_mask = encode_text(
            title, tokenizer, max_news_title_length, replace_first_token=True
        )
        body_inputs.append(body_input_ids)
        body_masks.append(body_mask)
        title_inputs.append(title_input_ids)
        title_masks.append(title_mask)

    pretrain_samples = pd.DataFrame({
        'body_inputs': body_inputs,
        'body_masks': body_masks,
        'title_inputs': title_inputs,
        'title_masks': title_masks
    })
    pretrain_samples.to_feather(os.path.join(processed_data_path, "pretrain.feather"))
    logging.info(f"Pre-training data processing completed: {len(pretrain_samples)} samples")


if __name__ == "__main__":
    # If simplified file does not exist
    if not os.path.exists(news_file_path_sp):
        logging.info(f"Simplified data does not exist: {news_file_path_sp}")
        logging.info("Generating simplified data")
        news = pd.read_csv(news_file_path, sep='\t')
        news[['News ID', 'Headline', 'News body']].to_csv(news_file_path_sp, sep='\t', index=False)
        logging.info("Simplified data generation completed")

    main()
