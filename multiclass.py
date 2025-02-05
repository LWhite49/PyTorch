import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

NUM_CLASSES = 4
INPUT_FEATURES = 2
SEED = 12

# Set sed and create data blobs

torch.manual_seed(SEED)

x, y = make_blobs(n_samples=1000, centers=NUM_CLASSES, n_features=INPUT_FEATURES) # Features for X, labels for Y

# Create Tensors

x_tensor = torch.from_numpy(x).type(torch.float32)
y_tensor = torch.from_numpy(y).type(torch.long)

# Split to train and test sets

break_point = int(len(x) * 0.8)

x_train, x_test = x_tensor[:break_point], x_tensor[break_point:]
y_train, y_test = y_tensor[:break_point], y_tensor[break_point:]

# Create Model

class MultiClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=INPUT_FEATURES, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=NUM_CLASSES)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        return self.layer3(torch.relu(self.layer2(torch.relu(self.layer1(x)))))
    


# Set up model, loss function, and optimizer

model = MultiClass()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Training loop

epochs = 300

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}")
            print(f"Accuracy: {accuracy_score(y_test, torch.argmax(test_pred, dim=1))}")


# Make final predictions

model.eval()
with torch.inference_mode():
    final_pred = model(x_test)
    final_pred = torch.argmax(final_pred, dim=1)

# Plot data

plt.scatter(x_test[:, 0], x_test[:, 1], c=final_pred.detach().numpy())
plt.show()

# Print final accuracy
print(f"Final Accuracy: {accuracy_score(y_test, final_pred)}")


