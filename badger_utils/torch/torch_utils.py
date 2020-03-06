import torch


def id_to_one_hot(data: torch.Tensor, vector_len: int):
    """Converts ID to one-hot representation.
    Each element in `data` is converted into a one-hot-representation - a vector of
    length vector_len having all zeros and one 1 on the position of value of the element.
    Args:
        data: ID of a class, it must hold for each ID that 0 <= ID < vector_len
        vector_len: length of the output tensor for each one-hot-representation
        dtype: data type of the output tensor
    Returns:
        Tensor of size [data.shape[0], vector_len] with one-hot encoding.
        For example, it converts the integer cluster indices of size [flock_size, batch_size] into
        one hot representation [flock_size, batch_size, n_cluster_centers].
    """
    device = data.device
    data_a = data.view(-1, 1)
    n_samples = data_a.shape[0]
    output = torch.zeros(n_samples, vector_len, device=device)
    output.scatter_(1, data_a, 1)
    output_dims = data.size() + (vector_len,)
    return output.view(output_dims)
