import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weight = -5
bias = 3
start = -7
end = 11
step = 0.1

X = torch.arange(start, end, step)
Y = torch.sin(X + weight) + bias
X_split = int(0.7 * len(X))
X_train, Y_train = X[:X_split], Y[:X_split]
X_test, Y_test = X[X_split:], Y[X_split:]

def plot_predictions(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, predictions=None):
    plt.figure(figsize=(10,5))
    plt.scatter(X, Y, c="b", label="Sine")
    plt.scatter(X_test, Y_test, c="g", label="Test")
    if(predictions is not None):
        plt.scatter(X_test, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()

#plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x + self.weight) + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()
with torch.inference_mode():
    y_preds = model_0.forward(X_test)
print(list(model_0.parameters()))
print(model_0.state_dict())
print(y_preds)
#plot_predictions(predictions=y_preds)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
torch.manual_seed(42)

epoches = 101
train_loss_value = []
test_loss_value = []
epoch_count = []

for epoch in range(epoches):
    model_0.train()
    y_pred = model_0.forward(X_test)
    train_loss = loss_fn(y_pred, Y_test)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0.forward(X_test)
        test_loss = loss_fn(test_pred, Y_test)
    
        if epoch % 10 == 0:
            train_loss_value.append(train_loss.detach().numpy())
            test_loss_value.append(test_loss.detach().numpy())
            epoch_count.append(epoch)
            print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")

def scatter_prediction(epoch_count=epoch_count, train_loss_value=train_loss_value, test_loss_value=test_loss_value):
    plt.figure(figsize=(10,5))
    plt.plot(epoch_count, train_loss_value, label="Train Loss")
    plt.plot(epoch_count, test_loss_value, label="Test Loss")
    plt.title("Training and Test Loss Curves")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

print(f"This model learned the following value for weight and bias:")
print(model_0.state_dict())
print("\nAnd the original value for weight and bias are:")
print(f"Weight: {weight} | Bias: {bias}")
#scatter_prediction()

print(y_preds)
model_0.eval()
with torch.inference_mode():
    y_preds = model_0.forward(X_test)
print(y_preds)    
#plot_predictions(predictions=y_preds)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "02_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0.forward(X_test)

print(y_preds == loaded_model_preds)
print(list(loaded_model_0.parameters()))
