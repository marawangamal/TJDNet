import torch
from tjdnet.tensorops.ccp import select_margin_ccp_tensor_batched


def main():
    batch_size, rank, seq_len, vocab_size = 8, 32, 32, 512
    cp_params = torch.ones(batch_size, rank, seq_len, vocab_size)  # (R, T, D)
    ops = torch.zeros(batch_size, seq_len // 2, dtype=torch.int64)
    ops[:, -1] = -1  # free leg
    ops = torch.cat(
        [ops, torch.ones(batch_size, seq_len // 2, dtype=torch.int64) * -2], dim=1
    )
    result_batched, _ = select_margin_ccp_tensor_batched(
        cp_params,
        cp_decode=torch.eye(vocab_size, vocab_size)
        .unsqueeze(0)
        .expand(batch_size, -1, -1),
        ops=ops,
    )  # (rank, n_free, vocab_size)


if __name__ == "__main__":
    main()
