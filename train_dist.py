import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
BATCH_SIZE = 1000
VOCAB_SIZE = 5
HORIZON = 3
EMBEDDING_DIM = 6
HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.01
NUM_EPOCHS = 200

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

    def forward(self, x):
        h = self.backbone(x)
        logits = [head(h) for head in self.heads]  # List of (B, V)
        logits = torch.stack(logits, dim=1)  # (B, H, V)
        return logits

    def sample(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)  # (B, H)


# Data
train_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
train_y = generate_y_from_x(train_x)
test_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, EMBEDDING_DIM)).float()
test_y = generate_y_from_x(test_x)

# Model, optimizer, loss
model = SimpleMultiHeadModel(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, HORIZON, VOCAB_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(train_x)  # (B, H, V)
    loss = torch.stack(
        [criterion(logits[:, i, :], train_y[:, i]) for i in range(HORIZON)]
    ).mean()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    pred = model.sample(test_x)
    accuracy = (pred == test_y).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")

print("Done!")
