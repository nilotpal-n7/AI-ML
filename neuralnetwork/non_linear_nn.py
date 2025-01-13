from linear_nn import accuraccy_fn, plotter, device, x_train, x_test, y_train, y_test
import matplotlib.pyplot as plt
from torch import nn
import torch

def plot_pred(epoch_count, train_loss_value, test_loss_value):
    plt.figure(figsize=(10,5))
    plt.plot(epoch_count, train_loss_value, label="Train Loss")
    plt.plot(epoch_count, test_loss_value, label="Test Loss")
    plt.title("Training and Test Loss Curves")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

class CircleModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
model_2 = CircleModel_v2().to(device)
print(model_2)
print(list(model_2.parameters()))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)
epoches = 201
train_loss_value = []
test_loss_value = []
epoch_count = []

for epoch in range(epoches):
    model_2.train()
    y_logits = model_2(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    train_loss = loss_fn(y_logits, y_train)
    train_acc = (y_train, y_pred)
    optimizer.zero_grad
    train_loss.backward()
    optimizer.step()

    model_2.eval()
    test_logits = model_2(x_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuraccy_fn(y_test, test_pred)

    if epoch % 100 == 0:
        train_loss_value.append(train_loss.detach().numpy())
        test_loss_value.append(test_loss.detach().numpy())
        epoch_count.append(epoch)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Acc.: {test_acc*100:.3f}% | Test Loss: {test_loss:.5f}, Test Acc.: {test_acc*100:.3f}%")

model_2.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_2(x_test))).squeeze()
#plotter(model_2)
#plot_pred(epoch_count, train_loss_value, test_loss_value)

a = torch.arange(-10, 10, 1, dtype=torch.float32)
print(a)
plt.plot(a, c="b", label="a")

def relu(x):
    return torch.max(torch.tensor(0), x)
print(relu(a))
plt.plot(relu(a), c="g", label="relu(a)")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
print(sigmoid(a))
plt.plot(sigmoid(a), c="r", label="sigmoid(a)")
plt.legend()
plt.show()
