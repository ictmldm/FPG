import pandas as pd
import numpy as np
import torch
from config.run_config import batch_size
from torch.utils.data import TensorDataset, DataLoader


def pretrain_data(file_path, batch_size=batch_size):
    """返回用于fine-tune Bart的预训练数据加载器"""
    pretrain = pd.read_feather(file_path)
    body_inputs = torch.tensor(np.array(pretrain['body_inputs'].tolist()).astype(int))
    body_masks = torch.tensor(np.array(pretrain['body_masks'].tolist()).astype(int))
    title_inputs = torch.tensor(np.array(pretrain['title_inputs'].tolist()).astype(int))
    title_masks = torch.tensor(np.array(pretrain['title_masks'].tolist()).astype(int))

    target = torch.ones_like(title_inputs)
    target[:, :-1] = title_inputs[:, 1:]
    target[target == 1] = -100  # 忽略填充部分

    dataset = TensorDataset(body_inputs, body_masks, title_inputs, title_masks, target)
    pretrain_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return pretrain_dataloader


def pretrain_test_data(file_path, batch_size=batch_size):
    """返回用于测试的预训练数据加载器"""
    test = pd.read_feather(file_path)
    body_inputs = torch.tensor(np.array(test['body_inputs'].tolist()).astype(int))
    body_masks = torch.tensor(np.array(test['body_masks'].tolist()).astype(int))
    titles = test['titles'].tolist()

    dataset = TensorDataset(body_inputs, body_masks)
    pretrain_test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return pretrain_test_dataloader, titles


def training_data(file_path, batch_size=batch_size):
    """返回用于训练步骤1的数据加载器"""
    train = pd.read_feather(file_path)
    
    train['h_inputs'] = train['h_inputs'].apply(lambda x: np.array(list(x)))
    # h_inputs = torch.from_numpy(np.array(train['h_inputs'].values.tolist()).astype(int))
    h_inputs = torch.tensor(np.array(train['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(train['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(train['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(train['bodys_masks'].tolist()).astype(int))
    titles = torch.tensor(np.array(train['titles'].tolist()).astype(int))
    titles_masks = torch.tensor(np.array(train['titles_masks'].tolist()).astype(int))

    target = torch.ones_like(titles)
    target[:, :-1] = titles[:, 1:]
    target[target == 1] = -100  # 忽略填充部分

    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, target)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def cl_training_data(file_path, batch_size=int(batch_size/2)):
    """返回用于训练步骤4的数据加载器"""
    train = pd.read_feather(file_path)
    
    # t1 = train['h_inputs'].values.tolist()
    # t2 = np.array(train['h_inputs'].values.tolist())
    # print(t2.shape)
    # t3 = np.array(train['h_inputs'].values.tolist()).astype(int)
    train['h_inputs'] = train['h_inputs'].apply(lambda x: np.array(list(x)))
    
    h_inputs = torch.tensor(np.array(train['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(train['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(train['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(train['bodys_masks'].tolist()).astype(int))
    p_titles = torch.tensor(np.array(train['pos_titles'].tolist()).astype(int))
    p_titles_masks = torch.tensor(np.array(train['pos_titles_masks'].tolist()).astype(int))
    n_titles = torch.tensor(np.array(train['neg_titles'].tolist()).astype(int))
    n_titles_masks = torch.tensor(np.array(train['neg_titles_masks'].tolist()).astype(int))

    p_target = torch.ones_like(p_titles)
    p_target[:, :-1] = p_titles[:, 1:]
    p_target[p_target == 1] = -100  # 忽略填充部分

    n_target = torch.ones_like(n_titles)
    n_target[:, :-1] = n_titles[:, 1:]
    n_target[n_target == 1] = -100  # 忽略填充部分

    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks, p_titles, p_titles_masks, n_titles, n_titles_masks, p_target, n_target)
    cl_train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return cl_train_dataloader


def test_data(file_path, batch_size=batch_size):
    """返回用于测试的数据加载器"""
    test = pd.read_feather(file_path)

    test['h_inputs'] = test['h_inputs'].map(lambda x: np.array(list(x)))
    h_inputs = torch.tensor(np.array(test['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(test['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(test['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(test['bodys_masks'].tolist()).astype(int))
    p_titles = test['p_titles'].tolist()

    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, p_titles


def small_test_data(file_path, batch_size=batch_size):
    """返回用于小规模测试的数据加载器"""
    test = pd.read_feather(file_path)

    test['h_inputs'] = test['h_inputs'].map(lambda x: np.array(list(x)))
    h_inputs = torch.tensor(np.array(test['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(test['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(test['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(test['bodys_masks'].tolist()).astype(int))
    p_titles = test['p_titles'].tolist()

    index = np.random.randint(0, len(p_titles), size=100 * batch_size)
    dataset = TensorDataset(h_inputs[index], h_masks[index], bodys[index], bodys_masks[index])
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, np.array(p_titles)[index]
