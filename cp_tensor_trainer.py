import torch
import torch.nn as nn
import tntorch as tn


# Simple NN that generates CP tensor factors
class CPTensorNet(nn.Module):
    def __init__(self, input_dim=5, tensor_shape=[3, 4, 2], rank=2):
        super().__init__()
        self.generators = nn.ModuleList(
            [nn.Linear(input_dim, shape * rank) for shape in tensor_shape]
        )
        self.tensor_shape = tensor_shape
        self.rank = rank

    def forward(self, x):
        factors = [
            gen(x).view(-1, shape, self.rank)
            for gen, shape in zip(self.generators, self.tensor_shape)
        ]
        return [tn.Tensor([f[i] for f in factors]) for i in range(x.size(0))]


# Simple loss based on tensor selection
def selection_loss(tensors, indices=[(0, 0, 0), (1, 1, 1)], targets=[1.0, 2.0]):
    loss = torch.tensor(0.0, requires_grad=True)
    targets_tensor = torch.tensor(targets)
    for tensor in tensors:
        selected = torch.tensor([tensor[idx].item() for idx in indices])
        loss = loss + nn.functional.mse_loss(selected, targets_tensor)
    return loss / len(tensors)


# Training
model = CPTensorNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    x = torch.randn(4, 5)
    tensors = model(x)
    loss = selection_loss(tensors)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training done!")
