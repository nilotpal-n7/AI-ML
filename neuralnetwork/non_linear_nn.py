from linear_nn import accuracy_fn, plotter, device, x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
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
    train_acc = accuracy_fn(y_train, y_pred)
    optimizer.zero_grad
    train_loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

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
#plt.show()



x_blob, y_blob = make_blobs(n_samples=100, n_features=2, centers=5, cluster_std=1.5, random_state=42)
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob, test_size=0.2, random_state=42)

print(x_blob[:5], y_blob[:5])
plt.figure(figsize=(10, 7))
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.legend()
#plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_3 = BlobModel(input_features=2, output_features=5, hidden_units=8).to(device)
print(model_3)
print(list(model_3.parameters()))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)

y_logits = model_3(x_blob_train.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])
print(torch.sum(y_pred_probs[0]))
print(y_pred_probs.argmax(dim=1))

torch.manual_seed(42)
epoches = 101
x_blob_train, y_blob_train = x_blob_train.to(device), y_blob_train.to(device)
x_blob_test, y_blob_test = x_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epoches):
    model_3.train()
    y_logits = model_3(x_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    train_loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(x_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)
    
    if epoch % 10 == 0:
        train_loss_value.append(train_loss.detach().numpy())
        test_loss_value.append(test_loss.detach().numpy())
        epoch_count.append(epoch)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Acc.: {test_acc*100:.3f}% | Test Loss: {test_loss:.5f}, Test Acc.: {test_acc*100:.3f}%")

model_3.eval()
with torch.inference_mode():
    y_preds = torch.softmax(model_3(x_blob_test), dim=1).argmax(dim=1)
print(y_preds[:10])
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")
#plotter(model_3, model_3, x_blob_train, y_blob_train, x_blob_test, y_blob_test)
