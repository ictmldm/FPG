import pandas as pd
import numpy as np
import torch
from config.run_config import batch_size
from torch.utils.data import TensorDataset, DataLoader
def pretrain_data(file_path, batch_size=batch_size):
    """
    Returns a data loader for pre-training Bart for fine-tuning.

    Args:
        file_path (str): Path to the feather file containing pre-training data.
        batch_size (int): Batch size for the data loader.

    Returns:
        DataLoader: DataLoader for pre-training data.
    """
    pretrain = pd.read_feather(file_path)

    # Convert list columns to numpy arrays and then to torch tensors
    body_inputs = torch.tensor(np.array(pretrain['body_inputs'].tolist()).astype(int))
    body_masks = torch.tensor(np.array(pretrain['body_masks'].tolist()).astype(int))
    title_inputs = torch.tensor(np.array(pretrain['title_inputs'].tolist()).astype(int))
    title_masks = torch.tensor(np.array(pretrain['title_masks'].tolist()).astype(int))

    # Create target tensor for language modeling (shifted title inputs)
    target = torch.ones_like(title_inputs)
    target[:, :-1] = title_inputs[:, 1:]
    # Ignore padding tokens (represented by 1 in this case) in the loss calculation
    target[target == 1] = -100

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(body_inputs, body_masks, title_inputs, title_masks, target)
    pretrain_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return pretrain_dataloader


def pretrain_test_data(file_path, batch_size=batch_size):
    """
    Returns a data loader and original titles for pre-training testing.

    Args:
        file_path (str): Path to the feather file containing pre-training test data.
        batch_size (int): Batch size for the data loader.

    Returns:
        tuple: A tuple containing:
            - DataLoader: DataLoader for pre-training test data.
            - list: List of original titles.
    """
    test = pd.read_feather(file_path)

    # Convert list columns to numpy arrays and then to torch tensors
    body_inputs = torch.tensor(np.array(test['body_inputs'].tolist()).astype(int))
    body_masks = torch.tensor(np.array(test['body_masks'].tolist()).astype(int))
    # Get original titles as a list
    titles = test['titles'].tolist()

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(body_inputs, body_masks)
    pretrain_test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return pretrain_test_dataloader, titles


def training_data(file_path, batch_size=batch_size):
    """
    Returns a data loader for training step 1.

    Args:
        file_path (str): Path to the feather file containing training data.
        batch_size (int): Batch size for the data loader.

    Returns:
        DataLoader: DataLoader for training data.
    """
    train = pd.read_feather(file_path)

    # Apply function to convert string representation of list to numpy array
    train['h_inputs'] = train['h_inputs'].apply(lambda x: np.array(list(x)))
    # Convert numpy arrays to torch tensors
    h_inputs = torch.tensor(np.array(train['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(train['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(train['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(train['bodys_masks'].tolist()).astype(int))
    titles = torch.tensor(np.array(train['titles'].tolist()).astype(int))
    titles_masks = torch.tensor(np.array(train['titles_masks'].tolist()).astype(int))

    # Create target tensor for language modeling (shifted titles)
    target = torch.ones_like(titles)
    target[:, :-1] = titles[:, 1:]
    # Ignore padding tokens (represented by 1 in this case) in the loss calculation
    target[target == 1] = -100

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks, titles, titles_masks, target)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def cl_training_data(file_path, batch_size=int(batch_size/2)):
    """
    Returns a data loader for training step 4 (Contrastive Learning).

    Args:
        file_path (str): Path to the feather file containing CL training data.
        batch_size (int): Batch size for the data loader (half of the standard batch size).

    Returns:
        DataLoader: DataLoader for CL training data.
    """
    train = pd.read_feather(file_path)

    # Apply function to convert string representation of list to numpy array
    train['h_inputs'] = train['h_inputs'].apply(lambda x: np.array(list(x)))

    # Convert numpy arrays to torch tensors
    h_inputs = torch.tensor(np.array(train['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(train['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(train['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(train['bodys_masks'].tolist()).astype(int))
    p_titles = torch.tensor(np.array(train['pos_titles'].tolist()).astype(int))
    p_titles_masks = torch.tensor(np.array(train['pos_titles_masks'].tolist()).astype(int))
    n_titles = torch.tensor(np.array(train['neg_titles'].tolist()).astype(int))
    n_titles_masks = torch.tensor(np.array(train['neg_titles_masks'].tolist()).astype(int))

    # Create target tensors for positive and negative titles (shifted inputs)
    p_target = torch.ones_like(p_titles)
    p_target[:, :-1] = p_titles[:, 1:]
    # Ignore padding tokens (represented by 1 in this case)
    p_target[p_target == 1] = -100

    n_target = torch.ones_like(n_titles)
    n_target[:, :-1] = n_titles[:, 1:]
    # Ignore padding tokens (represented by 1 in this case)
    n_target[n_target == 1] = -100

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks, p_titles, p_titles_masks, n_titles, n_titles_masks, p_target, n_target)
    cl_train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return cl_train_dataloader


def test_data(file_path, batch_size=batch_size):
    """
    Returns a data loader and original positive titles for testing.

    Args:
        file_path (str): Path to the feather file containing test data.
        batch_size (int): Batch size for the data loader.

    Returns:
        tuple: A tuple containing:
            - DataLoader: DataLoader for test data.
            - list: List of original positive titles.
    """
    test = pd.read_feather(file_path)

    # Apply function to convert string representation of list to numpy array
    test['h_inputs'] = test['h_inputs'].map(lambda x: np.array(list(x)))
    # Convert numpy arrays to torch tensors
    h_inputs = torch.tensor(np.array(test['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(test['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(test['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(test['bodys_masks'].tolist()).astype(int))
    # Get original positive titles as a list
    p_titles = test['p_titles'].tolist()

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(h_inputs, h_masks, bodys, bodys_masks)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, p_titles


def small_test_data(file_path, batch_size=batch_size):
    """
    Returns a data loader and a small subset of original positive titles for small-scale testing.

    Args:
        file_path (str): Path to the feather file containing test data.
        batch_size (int): Batch size for the data loader.

    Returns:
        tuple: A tuple containing:
            - DataLoader: DataLoader for a small subset of test data.
            - numpy.ndarray: Numpy array of a small subset of original positive titles.
    """
    test = pd.read_feather(file_path)

    # Apply function to convert string representation of list to numpy array
    test['h_inputs'] = test['h_inputs'].map(lambda x: np.array(list(x)))
    # Convert numpy arrays to torch tensors
    h_inputs = torch.tensor(np.array(test['h_inputs'].tolist()).astype(int))
    h_masks = torch.tensor(np.array(test['h_masks'].tolist()).astype(int))
    bodys = torch.tensor(np.array(test['bodys'].tolist()).astype(int))
    bodys_masks = torch.tensor(np.array(test['bodys_masks'].tolist()).astype(int))
    # Get original positive titles as a list
    p_titles = test['p_titles'].tolist()

    # Select a random subset of data
    index = np.random.randint(0, len(p_titles), size=100 * batch_size)
    # Create a TensorDataset and DataLoader for the subset
    dataset = TensorDataset(h_inputs[index], h_masks[index], bodys[index], bodys_masks[index])
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, np.array(p_titles)[index]
