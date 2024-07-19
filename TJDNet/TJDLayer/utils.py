from typing import Optional, Dict, List
import torch


def create_core_ident(batch_size: int, vocab_size: int, rank: int) -> torch.Tensor:
    """
    Create a core identity tensor for use in tensor operations.

    Parameters:
    - batch_size (int): Number of samples in a batch.
    - vocab_size (int): Size of the vocabulary or the dimension to repeat identity matrices.
    - rank (int): Size of the identity matrix (rank x rank).

    Returns:
    - torch.Tensor: A tensor of shape (batch_size, rank, vocab_size, rank) where each sub-matrix (rank, rank)
      along the vocab_size dimension is an identity matrix, repeated for each batch.
    """
    # Create an identity matrix of size rank x rank
    identity = torch.eye(rank, rank)

    # Expand this identity matrix to the required dimensions
    # The identity matrix needs to be replicated across the vocab_size and then batch_size dimensions
    # This requires the identity matrix to first be in the shape (1, rank, 1, rank) to align with the final target shape
    core_ident = identity.unsqueeze(0).unsqueeze(2)  # Shape: (1, rank, 1, rank)

    # Now repeat the structure to match the batch_size and vocab_size
    core_ident = core_ident.repeat(
        batch_size, 1, vocab_size, 1
    )  # Shape: (batch_size, rank, vocab_size, rank)

    return core_ident


def apply_id_transform(target, id_map):
    """
    Apply an ID transformation to a tensor of token IDs based on a provided mapping dictionary.
    This version supports negative indices and is implemented in a more Pythonic way.

    Parameters:
    - target (torch.Tensor): The tensor of token IDs to transform.
    - id_map (dict): A dictionary where keys are original token IDs and values are the transformed token IDs.

    Returns:
    - torch.Tensor: A tensor with the same shape as target, where each token ID has been transformed according to id_map.
    """
    # Use a list comprehension to apply the transformation to each element
    # This handles any range of indices correctly including negatives
    if not id_map:
        return target

    transformed_list = [
        [id_map.get(int(id), id) for id in row] for row in target.tolist()
    ]
    return torch.tensor(transformed_list, dtype=target.dtype, device=target.device)


# Test the function as described earlier in the test setup.
def sample_from_tensor_dist(dist: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample from a tensor distribution.

    Args:
        dist (torch.Tensor): Tensor of probabilities to sample from. Shape: (D1, D2, ..., DN).
        n_samples (int): Number of samples to draw from the distribution.

    Returns:
        torch.Tensor: Samples drawn from the distribution. Shape: (n_samples, N).
    """

    # Flatten the probability tensor
    flattened_dist = dist.view(-1)

    # Sample indices from the flattened distribution
    sampled_indices = torch.multinomial(flattened_dist, n_samples, replacement=True)

    # Convert the flat indices back to the original multi-dimensional indices
    sampled_multi_indices = torch.unravel_index(sampled_indices, dist.shape)

    # Stack the multi-dimensional indices to get the final samples
    samples = torch.stack(sampled_multi_indices, dim=-1)

    return samples


def batched_index_select(
    input: torch.Tensor,
    batched_index: torch.Tensor,
):
    """Perform a batched index select operation.

    Args:
        input (torch.Tensor): Input tensor. Shape: (d1, d2, ..., dN).
        batched_index (torch.Tensor): Batched index tensor. Shape: (batch_size, N).

    Returns:
        torch.Tensor: Output tensor. Shape: (batch_size, d2, ..., dN).
    """
    cols = [batched_index[:, i] for i in range(batched_index.shape[1])]
    return input[cols]


def check_naninf(x: torch.Tensor, msg: Optional[str] = None, raise_error: bool = True):
    if torch.isnan(x).any() or torch.isinf(x).any():
        if raise_error:
            raise ValueError(f"NaN/Inf detected in tensor: {msg}")
        else:
            return True


def select_and_marginalize_uMPS(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    n_core_repititions: int,
    selection_ids: Dict[int, int],
    marginalize_ids: List[int],
):
    """Given a uMPS, perform select and/or marginalize operations.

    Args:
        alpha (torch.Tensor): _description_
        beta (torch.Tensor): _description_
        core (torch.Tensor): _description_
        selection_ids (List[int]): _description_
        marginalize_ids (List[int]): _description_

    Returns:
        _description_

    """

    # Validation
    assert len(alpha.shape) == 1, "Alpha should be a 1D tensor"
    assert len(beta.shape) == 1, "Beta should be a 1D tensor"

    # Can't have same index in both selection and marginalization
    assert not any(
        [sid in marginalize_ids for sid in selection_ids.values()]
    ), "Can't have same index in both selection and marginalization"

    # Can't have indices out of range
    assert all(
        [sid < n_core_repititions for sid in selection_ids.values()]
    ), "Selection index out of range"

    assert all(
        [mid < n_core_repititions for mid in marginalize_ids]
    ), "Marginalization index out of range"

    result = None
    core_margin = torch.einsum(
        "idj,d->ij", core, torch.ones(core.shape[1], device=core.device)
    )
    for i in range(n_core_repititions):
        if i in selection_ids:
            sid = selection_ids[i]
            node = core[:, sid, :]

        elif i in marginalize_ids:
            node = core_margin
        else:
            node = core

        if result is None:
            result = node
        elif len(node.shape) == 2:
            shape_init = result.shape
            result = result.reshape(-1, shape_init[-1]) @ node
            result = result.reshape(tuple(shape_init[:-1]) + tuple(node.shape[1:]))
        else:
            shape_init = result.shape
            result_tmp = result.reshape(-1, shape_init[-1])
            result = torch.einsum("ij,jdl->idl", result_tmp, node)
            result = result.reshape(tuple(shape_init[:-1]) + tuple(node.shape[1:]))

    # Contract with alpha and beta
    if result is not None:
        shape_init = result.shape
        result = result.reshape(shape_init[0], -1, shape_init[-1])
        result = torch.einsum("i,idj,j->d", alpha, result, beta)
        result = result.reshape(tuple(shape_init[1:-1]))

    return result
