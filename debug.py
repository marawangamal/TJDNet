import torch
import sys
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched_w_decoder

path = sys.argv[1]
data = torch.load(path, map_location="cpu")
params = data["params"]
y = data["y"]

B, R, H, d = params.shape
ops = y.reshape(B, H)


def describe_tensor_table(tensor_list):
    print(
        f"{'Name':<12} | {'Shape':<20} | {'Min':>10} | {'Max':>10} | {'NaN':>13} | {'Inf':>13}"
    )
    print("-" * 80)
    for name, tensor in tensor_list:
        if torch.is_tensor(tensor):
            shape = tuple(tensor.shape)
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            total = tensor.numel()
            print(
                f"{name:<12} | {str(shape):<20} | {min_val:>10.3f} | {max_val:>10.3f} | {nan_count:>6}/{total:<8} | {inf_count:>6}/{total:<8}"
            )
        else:
            print(f"{name:<12} | {'not a tensor':<20}")


describe_tensor_table(list(data.items()))


p_tilde, _ = select_margin_cp_tensor_batched_w_decoder(
    cp_params=params,
    ops=ops,
    decoder=data["cp_decoder"],
)

print("NaN in p_tilde:", torch.isnan(p_tilde).any().item())
print("Inf in p_tilde:", torch.isinf(p_tilde).any().item())
