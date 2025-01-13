from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
import pandas as pd
import requests
import torch

n_samples = 1000
x, y = make_circles(n_samples, noise=0.03, random_state=42)
print(f"First 5 X features:\n{x[:5]}")
print(f"First 5 Y lables:\n{y[:5]}")

circles = pd.DataFrame({"x1": x[:, 0], "x2": x[:,1], "label": y})
print(circles.head(10))
print(circles.label.value_counts())
'''plt.scatter(x=x[:,0], y=x[:,1], c=y, cmap=plt.cm.RdYlBu, label="Test")
plt.legend()
#plt.show()'''

print(x.shape, y.shape)
x_sample = x[0]
y_sample = y[0]
print(f"Values for one sample of x: {x_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of x: {x_sample.shape} and the same for y: {y_sample.shape}")

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(x[:5], y[:5])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(x_train), len(x_test), len(y_train), len(x_test))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

def accuraccy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred))
    return acc

def downloader(file_name):
    if Path(file_name).is_file():
        print(f"{file_name} already exist, skipping download")
    else:
        print(f"Downloading {file_name}")
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        with open(f"{file_name}", "wb") as f:
            f.write(request.content)

def plotter(model):
    downloader("helper_functions.py")
    from helper_functions import plot_predictions, plot_decision_boundary

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.legend()
    plt.show()

class CircleModel_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
        
model_0 = CircleModel_v0().to(device)
print(model_0)
print(list(model_0.parameters()))

untrained_preds = model_0(x_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

#model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5), nn.Linear(in_features=5, out_features=1)).to(device)
#print(model_0)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

y_logits = model_0(x_test.to(device))[:5]
print(y_logits)
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(x_test.to(device))[:5]))
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
y_preds.squeeze()

torch.manual_seed(42)
epoches = 101

for epoch in range(epoches):
    model_0.train()
    y_logits = model_0(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    train_loss = loss_fn(y_logits, y_train)
    train_acc = accuraccy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuraccy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Acc.: {test_acc*100:.2f}% | Test Loss: {test_loss:.5f}, Test Acc.: {test_acc*100:.2f}%")

'''model_0.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_0(x_test))).squeeze()
plotter(model_0)'''

class CircleModel_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_1 = CircleModel_v1().to(device)
print(model_1)
print(list(model_1.parameters()))
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

torch.manual_seed(42)
epoches = 1001
for epoch in range(epoches):
    model_1.train()
    y_logits = model_1(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    train_loss = loss_fn(y_logits, y_train)
    train_acc = accuraccy_fn(y_train, y_pred)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_1.eval()
    test_logits = model_1(x_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuraccy_fn(y_test, test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Acc.: {test_acc*100:.2f}% | Test Loss: {test_loss:.5f}, Test Acc.: {test_acc*100:.2f}%")

'''model_1.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_1(x_test))).squeeze()
plotter(model_1)'''

n_samples = 1000
x, y = make_circles(n_samples, noise=0.03, random_state=42)
'''plt.scatter(x=x[:,0], y=x[:,1], c=y, cmap=plt.cm.RdYlBu, label="Test")
plt.legend()
#plt.show()'''
