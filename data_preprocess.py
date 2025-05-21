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

# 配置日志
log_filename = f"./logs/data_preprocessing.log"
# _{datetime.now().strftime('%Y%m%d_%H%M%S')}
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 检查并创建保存处理数据的目录
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)
    logging.info(f"创建目录: {processed_data_path}")


def load_data(file_path, data_type='news'):
    """加载数据"""
    logging.info(f"加载{data_type}数据: {file_path}")
    data = pd.read_csv(file_path, sep='\t')
    data.fillna(value=" ", inplace=True)
    logging.info(f"{data_type}数据加载完成")
    return data


def prepare_tokenizer(bart_name):
    """准备Bart分词器"""
    logging.info(f"加载Bart分词器: {bart_name}")
    tokenizer = BartTokenizer.from_pretrained(
        bart_name,
        trust_remote_code=True,
        local_files_only=True,
        do_lower_case=True
    )
    logging.info("Bart分词器加载完成")
    return tokenizer


def build_news_dict(news_ids):
    """构建新闻索引字典"""
    logging.info("构建新闻索引字典")
    news_dict = {news_id: idx + 1 for idx, news_id in enumerate(news_ids)}
    logging.info("新闻索引字典构建完成")
    return news_dict


def process_test_data(test, news_titles, news_bodys, news_dict, tokenizer,
                     max_click_length, max_news_title_length, max_news_body_length):
    """处理测试数据"""
    logging.info("开始处理测试数据")
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

    print("test_ids: {}".format(len(test_ids)))
    logging.info("测试数据处理完成")
    logging.info(f"测试数据ID总数: {len(test_ids)}")
    test_samples = pd.DataFrame({
        'h_inputs': h_inputs,
        'h_masks': h_masks,
        'bodys': bodys,
        'bodys_masks': bodys_masks,
        'p_titles': p_titles
    })
    print("test_logs:{}".format(len(test_samples)))
    test_samples.to_feather(os.path.join(processed_data_path, 'test.feather'))
    logging.info(f"测试数据保存完成，共 {len(test_samples)} 样本")

    return test_samples, test_ids  # Return test_ids for exclusion


def process_click_history(click_history_ids, news_titles, news_dict, tokenizer,
                         max_click_length, data_type="train"):
    """处理点击历史"""
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
    """编码文本"""
    text_encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    if replace_first_token:
        text_encoded['input_ids'][0][0] = tokenizer.eos_token_id
    return text_encoded['input_ids'].squeeze(0).tolist(), text_encoded['attention_mask'].squeeze(0).tolist()


def process_samples(samples, news_titles, news_bodys, news_dict, tokenizer,
                   max_click_length, max_news_title_length, max_news_body_length, limit=None):
    """处理样本数据"""
    logging.info("开始处理样本数据")
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

    logging.info("样本数据处理完成")
    return h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, news_count, user_count


def process_raw_test(test, news_titles, news_bodys, news_dict, tokenizer, max_click_length):
    """处理测试数据"""
    logging.info("开始处理raw测试数据")
    h_inputs, bodys, o_titles, p_titles, test_ids = [], [], [], [], []

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
    logging.info("raw测试数据处理完成")
    return test_samples, test_ids


def save_samples_to_feather(file_name, h_inputs, h_masks, bodys, bodys_masks, titles=None, titles_masks=None):
    """保存样本数据到Feather文件"""
    logging.info(f"保存样本数据到: {file_name}")
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
    logging.info(f"样本数据保存完成: {file_name}")


def main():
    logging.info("开始加载数据")
    news = load_data(news_file_path_sp)
    train = load_data(train_file_path, 'train')
    valid = load_data(dev_file_path, 'valid')
    test = load_data(test_file_path, 'test')

    news_ids, news_titles, news_bodys = news["News ID"].values, news["Headline"].values, news["News body"].values
    news_dict = build_news_dict(news_ids)
    tokenizer = prepare_tokenizer(bart_name)

    # 处理测试数据并获取测试用的新闻ID
    logging.info("处理测试数据")
    test_samples, test_ids = process_test_data(
        test, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length
    )

    # 处理训练数据并获取训练用的新闻ID
    h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, train_news_count, train_user_count = process_samples(
        train, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length, limit=limit
    )
    save_samples_to_feather(f"train_limit_to_{limit}.feather", h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks)
    logging.info(f"训练数据处理完成: {len(train)} 样本")
    logging.info(f"{len(train_news_count)} 新闻, {len(train_user_count)} 用户")

    # 处理验证数据
    logging.info("处理验证数据")
    h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, valid_news_count, valid_user_count = process_samples(
        valid, news_titles, news_bodys, news_dict, tokenizer,
        max_click_length, max_news_title_length, max_news_body_length
    )
    save_samples_to_feather("valid.feather", h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks)
    logging.info(f"验证数据处理完成: {len(valid)} 样本")
    logging.info(f"{len(valid_news_count)} 新闻, {len(valid_user_count)} 用户")

    # 收集所有需要排除的新闻ID（训练和测试）
    used_news_ids = set(train_news_count.keys()).union(set(test_ids))
    logging.info(f"预训练数据将排除 {len(used_news_ids)} 条新闻ID来自训练和测试数据")

    # 处理预训练数据
    logging.info("处理预训练数据")
    body_inputs, body_masks, title_inputs, title_masks = [], [], [], []
    for id in tqdm(news_ids, desc="Processing Pretrain Data"):
        # if id in used_news_ids:
        #     continue
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
    logging.info(f"预训练数据处理完成: {len(pretrain_samples)} 样本")


if __name__ == "__main__":
    # 如果简化文件不存在
    if not os.path.exists(news_file_path_sp):
        logging.info(f"简化数据不存在: {news_file_path_sp}")
        logging.info("正在生成简化数据")
        news = pd.read_csv(news_file_path, sep='\t')
        news[['News ID', 'Headline', 'News body']].to_csv(news_file_path_sp, sep='\t', index=False)
        logging.info("简化数据生成完成")

    main()
