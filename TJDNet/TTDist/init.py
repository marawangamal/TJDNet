import torch


def get_random_mps(batch_size, rank, vocab_size, dist="randn", *args, **kwargs):
    distrib_func = {"randn": torch.randn, "rand": torch.rand}[dist]
    alpha = distrib_func(1, rank).repeat(batch_size, 1).abs()
    beta = distrib_func(1, rank).repeat(batch_size, 1).abs()
    core = torch.nn.Parameter(
        distrib_func(rank)
        .unsqueeze(1)
        .repeat(1, vocab_size, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


# def get_onehot_mps(batch_size: int, rank: int, vocab_size: int):
#     onehot_ids = [int(torch.randint(0, vocab_size, (1,)).item()) for _ in range(rank)]
#     alpha = torch.ones(1, rank).repeat(batch_size, 1)
#     beta = torch.ones(1, rank).repeat(batch_size, 1)
#     coreOneHot = torch.zeros(rank, vocab_size, rank)
#     for onehot_idx in onehot_ids:
#         coreOneHot[:, onehot_idx, :] = torch.eye(rank)
#     core = torch.nn.Parameter(coreOneHot.unsqueeze(0).repeat(batch_size, 1, 1, 1))
#     return alpha, beta, core


# def get_init_params_randn_positive(batch_size, rank, vocab_size, *args, **kwargs):
#     alpha = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
#     beta = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
#     core = torch.nn.Parameter(
#         torch.randn(rank, vocab_size, rank)
#         .abs()
#         .unsqueeze(0)
#         .repeat(batch_size, 1, 1, 1)
#     )
#     return alpha, beta, core


# def get_randn_old(batch_size, rank, vocab_size, *args, **kwargs):
#     alpha = (
#         torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(torch.tensor(1 / rank))
#     ).abs()
#     beta = (
#         torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(torch.tensor(1 / rank))
#     ).abs()
#     core = torch.nn.Parameter(
#         torch.eye(rank)
#         .unsqueeze(1)
#         .repeat(1, vocab_size, 1)
#         .unsqueeze(0)
#         .repeat(batch_size, 1, 1, 1)
#     )
#     return alpha, beta, core
