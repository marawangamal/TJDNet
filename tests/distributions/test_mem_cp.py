import torch
import gc
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.tpnet import TensorParamNetConfig


def mem_check(msg: str = "unknown"):
    print(f"MEM [{msg}]: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# --inp_seq_len 8 --out_seq_len 32


def test_cpdist_eval_memory():
    """Test memory usage specifically for evaluate_at_points_and_get_norm_consts"""
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Create test data with smaller dimensions
    batch_size = 1
    input_seq_len = 8
    embedding_size = 4096
    hidden_dim = 4096
    vocab_size = 32000
    horizon = 2  # [2, 3, 4],
    # rank = 2

    for rank in [8]:
        print(f"Testing rank: {rank}")

        mem_check("starting point")
        config = BaseDistConfig(
            vocab_size=vocab_size,
            horizon=horizon,
            rank=rank,
            param_net=TensorParamNetConfig(
                hidden_dim=hidden_dim,
                in_dim=embedding_size,
            ),
        )

        model_head = CPDist(config).cuda()
        mem_check("after model creation")

        # Create inputs
        hidden_states = torch.randn(
            batch_size, input_seq_len, embedding_size, device="cuda", requires_grad=True
        )
        targets = torch.randint(
            0, vocab_size, (batch_size, input_seq_len, horizon), device="cuda"
        )
        mem_check("after input creation")

        # Run the key method
        mem_check("before evaluate_at_points_and_get_norm_consts")
        p_tilde, p_tilde_scale_factors, norm_const, norm_const_scale_factors = (
            model_head.evaluate_at_points_and_get_norm_consts(hidden_states, targets)
        )
        mem_check("after evaluate_at_points_and_get_norm_consts")

        # Compute loss
        mem_check("before loss computation")
        loss = (
            -torch.log(p_tilde)  # (B, T')
            + torch.log(norm_const)  # (B, T')
            # Contraction Stability Scale Factors
            - sum([torch.log(z) for z in p_tilde_scale_factors])  # (B, T')
            + sum([torch.log(z) for z in norm_const_scale_factors])
        )  # (B, T-H)
        mem_check("after loss computation")

        # Run backward pass
        mem_check("before backward")
        try:
            loss_rd = torch.mean(loss.sum(dim=-1))
            loss_rd.backward()
            mem_check("after backward")
            print("Backward pass completed successfully")
        except RuntimeError as e:
            current_mem = torch.cuda.memory_allocated() / 1e9
            print(f"OOM during backward at {current_mem:.2f} GB")
            print(f"Error: {e}")

        # Clean up
        del model_head, hidden_states, targets, p_tilde, norm_const, loss
        torch.cuda.empty_cache()

        # return "Test completed"


if __name__ == "__main__":
    test_cpdist_eval_memory()
