import torch
from torch import nn
import matplotlib.pyplot as plt

# Set seed
torch.manual_seed(12)
# Define Linear Regression Model -  All Models extend nn.Module
class LinearRegression(nn.Module):
    # Constructor specifies layers and parameters
    def __init__(self):
        super().__init__()
        # requires_grad=True means that the weights will be updated during training
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

    # Forward function specifies how input data is passed through the model - based on type of model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
    def log_state(self):
        print(f"Model state: {self.state_dict()}")
    

# Source input data
x_dataset = torch.arange(0,1,0.002).unsqueeze(1)
y_dataset = x_dataset * .7 + .3

#plt.scatter(x_dataset, y_dataset)
#plt.title("Data Distribution")
#plt.show()

# Partition data into training and testing sets
partition = int(len(x_dataset) * 0.8)
x_train = x_dataset[:partition]
y_train = y_dataset[:partition]
x_test = x_dataset[partition:]
y_test = y_dataset[partition:]

# Create model, loss function, and optimizer

model_0 = LinearRegression()

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Create training loop

epochs = 300

train_losses = []
test_losses = []
epoch_count = []

for epoch in range(epochs):
    # Put model in training mode
    model_0.train()

    # Forward Pass with training data
    y_pred = model_0(x_train)

    # Calculate Loss
    loss = loss_fn(y_pred, y_train)

    # Take Zero Gradient
    optimizer.zero_grad()

    # Backwards Pass
    loss.backward()

    # Update Parameters
    optimizer.step()

    # Put model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # Forward Pass test data
        test_pred = model_0(x_test)

        # Calculate loss from test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        # Log training state
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_losses.append(loss.detach().numpy())
            test_losses.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch}")
            model_0.log_state()

# Chart data with Matplotlib
print("Train Losses: ", train_losses)
print("Test Losses: ", test_losses)
plt.plot(epoch_count, train_losses, label="Train Loss")
plt.plot(epoch_count, test_losses, label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model_0.log_state()

# Make Predictions
model_0.eval()
with torch.inference_mode():
    y_pred = model_0(x_dataset)
    plt.plot(x_dataset, y_dataset, label="Actual")
    plt.plot(x_dataset, y_pred.detach().numpy(), label="Predicted", color='red')
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()





