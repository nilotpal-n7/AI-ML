import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

print(torch.__version__)

weight = 0.7
bias = 0.3
step = 0.02
start = 0
end = 1

X = torch.arange(start, end, step)
Y = weight * X + bias
train_split = int(0.8 * len(X))

X_train = X[:train_split]
Y_train = Y[:train_split]
X_test = X[train_split:]
Y_test = Y[train_split:]

def plot_prediction(train_data = X_train, train_labels = Y_train, test_data = X_test, test_labels = Y_test, prediction = None):
    plt.figure(figsize=(10, 10))

    plt.scatter(train_data, train_labels, c='b', label='Training data')
    plt.scatter(test_data, test_labels, c='g', label='Testing data')
    if prediction is not None:
        plt.scatter(test_data, prediction, c='r', label='Prediction')

    plt.legend(prop={'size': 15})
    plt.show()

#plot_prediction()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
#plot_prediction(prediction=y_preds)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
torch.manual_seed(42)

epoches = 201
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epoches):
    model_0.train()
    y_pred = model_0(X_train)
    train_loss = loss_fn(y_pred, Y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, Y_test.type(torch.float32))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(train_loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss} ")

plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epoches")
plt.legend()
plt.show()

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weight and bias are:")
print(f"Weight: {weight}, Bias: {bias}")

model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
plot_prediction(prediction=y_preds)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

print(y_preds == loaded_model_preds)
