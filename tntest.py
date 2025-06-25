import torch
import tntorch as tn

# Your setup
shape = [10, 20, 15]
rank = 5
factors = [torch.randn(shape[i], rank) for i in range(len(shape))]
cp_tensor = tn.Tensor([factors])

# Select specific indices
# Select index 3 from mode 0, index 7 from mode 1, keep all of mode 2
selected = cp_tensor[3, 7, :]

# Select index 5 from mode 0, keep all others
selected = cp_tensor[5, :, :]

# Select specific index from all modes
selected = cp_tensor[2, 10, 8]  # Returns a scalar
