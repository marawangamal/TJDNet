import torch
import psutil
import os
import gc
import resource


def get_memory_mb():
    """Get current memory usage in MB (CPU or CUDA if available)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_peak_memory_mb():
    """Get peak memory usage in MB (CPU or CUDA if available)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        if os.uname().sysname == "Darwin":
            return usage.ru_maxrss / 1024 / 1024  # bytes to MB
        else:
            return usage.ru_maxrss / 1024  # kB to MB


def problematic_method(B, d, R, T, D, device, mask_select, t):
    """Original memory-intensive approach"""
    # Memory blow-up: expands decoder to (B, d, D)
    decoder = torch.randn(d, D, device=device)
    cp_params = torch.randn(B, R, T, d, device=device)
    decoder_selected = torch.gather(
        decoder.unsqueeze(0).expand(mask_select.sum(), -1, -1),
        dim=-1,
        index=ops[mask_select, t].reshape(-1, 1, 1).expand(-1, decoder.shape[0], -1),
    ).squeeze(-1)

    return torch.einsum(
        "brd,bd -> br", cp_params[mask_select, :, t, :], decoder_selected
    )


def gather_method(B, d, R, T, D, device, mask_select, t):
    """Using gather on cp_params instead of decoder (no decoder multiplication)"""
    cp_params = torch.randn(B, R, T, D, device=device)  # Note: D instead of d
    update = torch.gather(
        cp_params[mask_select, :, t, :],  # (B', R, D) - note D, not d
        dim=-1,
        index=ops[mask_select, t].reshape(-1, 1, 1).expand(-1, R, -1),  # (B', R, 1)
    ).squeeze(-1)
    return update


def efficient_method(B, d, R, T, D, device, mask_select, t):
    """Memory-efficient approach"""
    # Direct indexing: no expansion needed
    decoder = torch.randn(d, D, device=device)
    cp_params = torch.randn(B, R, T, d, device=device)
    selected_indices = ops[mask_select, t]
    decoder_selected = decoder[:, selected_indices].T

    # Use bmm instead of einsum
    cp_chunk = cp_params[mask_select, :, t, :]
    return torch.bmm(cp_chunk, decoder_selected.unsqueeze(-1)).squeeze(-1)


# Test setup
B, d, R, D, T = 1000, 512, 64, 1024, 10
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# For decoder methods
ops = torch.randint(0, D, (B, T), device=device)

# For gather method (cp_params has vocab dimension)
mask_select = torch.rand(B, device=device) < 0.3
t = 5

# print(f"Selected: {mask_select.sum()}/{B} samples")
# print(
#     f"Decoder: ({d},{D}), cp_params_decoder: ({B},{R},10,{d}), cp_params_gather: ({B},{R},10,{D})"
# )

# Test memory usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()


def measure_peak_memory(method, *args):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = method(*args)
    mem_peak = get_peak_memory_mb()
    return result, mem_peak


results = []
memories = []

for method, name, args in [
    (
        problematic_method,
        "Problematic method (decoder*cp_params)",
        (B, d, R, T, D, device, mask_select, t),
    ),
    (
        gather_method,
        "Gather method (no decoder)",
        (B, d, R, T, D, device, mask_select, t),
    ),
    (
        efficient_method,
        "Efficient method (decoder*cp_params)",
        (B, d, R, T, D, device, mask_select, t),
    ),
]:
    result, mem = measure_peak_memory(method, *args)
    results.append((name, result))
    memories.append(mem)

print()
for i, (name, result) in enumerate(results):
    print(f"{name} peak memory: {memories[i]:.1f}MB")
