import torch
import torch.nn as nn
import torch.optim as optim
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPCond
from tjdnet.distributions.multihead import MultiHeadDist

# Hyperparameters
BATCH_SIZE = 1000
VOCAB_SIZE = 5
HORIZON = 3
EMBEDDING_DIM = 6
HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.01
NUM_EPOCHS = 500
RANK = 4

torch.manual_seed(42)


def generate_y_from_x(x, vocab_size=VOCAB_SIZE):
    """Generate y tokens from x using simple linear rules."""
    y0 = (x[:, 0] + x[:, 1]) % vocab_size
    y1 = (x[:, 2] + x[:, 3]) % vocab_size
    y2 = (x[:, 4] + x[:, 5]) % vocab_size
    return torch.stack([y0, y1, y2], dim=1).long()


class SimpleModel(nn.Module):
    """Simple model with backbone + distribution head."""

    def __init__(self, backbone, head_dist):
        super().__init__()
        self.backbone = backbone
        self.head = head_dist

    def forward(self, x, y):
        h = self.backbone(x)
        return self.head(h, y)

    def sample(self, x):
        h = self.backbone(x)
        sampled, _ = self.head.sample(h, lambda p: torch.argmax(p, dim=-1))
        return sampled


def create_backbone():
    """Create a simple backbone network."""
    layers = []
    for i in range(NUM_LAYERS):
        in_dim = EMBEDDING_DIM if i == 0 else HIDDEN_DIM
        layers.extend([nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU()])
    layers.append(nn.Tanh())  # Keep outputs bounded
    return nn.Sequential(*layers)


# Data
train_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
train_y = generate_y_from_x(train_x)
test_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
test_y = generate_y_from_x(test_x)

# Config
config = BaseDistConfig(
    vocab_size=VOCAB_SIZE, horizon=HORIZON, rank=RANK, embedding_dim=HIDDEN_DIM
)

# Models
models = {
    "MLP-MHead": SimpleModel(create_backbone(), MultiHeadDist(config)),
    "MLP-CP": SimpleModel(create_backbone(), CPDist(config)),
    "MLP-CPB": SimpleModel(create_backbone(), CPCond(config)),
}

optimizers = {
    name: optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for name, model in models.items()
}

# Pre-training evaluation
print("Pre-training accuracy...")
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        pred = model.sample(test_x)
        accuracy = (pred == test_y).float().mean()
        print(f"{name}: accuracy={accuracy:.4f}")

# Training
print("Training...")
for epoch in range(NUM_EPOCHS):
    losses = {}
    for name, model in models.items():
        model.train()
        optimizers[name].zero_grad()
        loss = model(train_x, train_y).mean()
        loss.backward()
        optimizers[name].step()
        losses[name] = loss.item()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch}: "
            + " ".join([f"{name}={loss:.4f}" for name, loss in losses.items()])
        )

# Final evaluation
print("\nTesting accuracy...")
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        pred = model.sample(test_x)
        accuracy = (pred == test_y).float().mean()
        print(f"{name}: accuracy={accuracy:.4f}")

print("Done!")
