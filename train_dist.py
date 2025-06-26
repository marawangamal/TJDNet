import torch
import torch.nn as nn
import torch.optim as optim
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPBDist
from tjdnet.distributions.cpe import CPEffDist

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
    y0 = (x[:, 0] + x[:, 1]) % vocab_size
    y1 = (x[:, 2] + x[:, 3]) % vocab_size
    y2 = (x[:, 4] + x[:, 5]) % vocab_size
    return torch.stack([y0, y1, y2], dim=1).long()


class SimpleMultiHeadModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, vocab_size):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(horizon)]
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        h = self.backbone(x)
        logits = [head(h) for head in self.heads]  # List of (B, V)
        logits = torch.stack(logits, dim=1)  # (B, H, V)
        # Compute mean cross-entropy loss over all positions
        loss = torch.stack(
            [self.criterion(logits[:, i, :], y[:, i]) for i in range(logits.size(1))]
        ).mean()
        return loss

    def sample(self, x):
        h = self.backbone(x)
        logits = [head(h) for head in self.heads]
        logits = torch.stack(logits, dim=1)
        return torch.argmax(logits, dim=-1)


class BackboneCPModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        horizon,
        vocab_size,
        rank,
        head_type="CP",
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)
        config = BaseDistConfig(
            vocab_size=vocab_size, horizon=horizon, rank=rank, embedding_dim=hidden_dim
        )
        if head_type == "CPB":
            self.head = CPBDist(config)
        elif head_type == "CPE":
            self.head = CPEffDist(config)
        else:
            self.head = CPDist(config)
        self.head_type = head_type

    def forward(self, x, y):
        h = self.backbone(x)
        return self.head(h, y)

    def sample(self, x):
        h = self.backbone(x)
        sampled, _ = self.head.sample(
            h, lambda p: torch.argmax(p, dim=-1), horizon=self.head.horizon
        )
        return sampled


# Data
train_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
train_y = generate_y_from_x(train_x)
test_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
test_y = generate_y_from_x(test_x)

# Models
models = {
    "MLP-Head": SimpleMultiHeadModel(
        EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, HORIZON, VOCAB_SIZE
    ),
    "MLP-CP": BackboneCPModel(
        EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, HORIZON, VOCAB_SIZE, RANK, head_type="CP"
    ),
    "MLP-CPB": BackboneCPModel(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        HORIZON,
        VOCAB_SIZE,
        RANK,
        head_type="CPB",
    ),
    # Fails
    # "MLP-CPE": BackboneCPModel(
    #     EMBEDDING_DIM,
    #     HIDDEN_DIM,
    #     NUM_LAYERS,
    #     HORIZON,
    #     VOCAB_SIZE,
    #     RANK,
    #     head_type="CPE",
    # ),
}

optimizers = {
    name: optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for name, model in models.items()
}

# Pre-training evaluation
print("\nPre-training accuracy...")
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

# Evaluation
print("\nTesting accuracy...")
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        pred = model.sample(test_x)
        accuracy = (pred == test_y).float().mean()
        print(f"{name}: accuracy={accuracy:.4f}")

print("Done!")
