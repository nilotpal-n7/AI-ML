import torch
import requests
import torchvision
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from torchvision.transforms import ToTensor

print(f"PyTorch Version: {torch.__version__}\nTorchVision Version: {torchvision.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

image, label = train_data[0]
print(image, label)
print(f"Imgae Shape: {image.shape}")
print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))
class_names = train_data.classes
print(class_names)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.legend()
#plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols +1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.legend()
#plt.show()

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
train_features_batch, train_labels_batch = next(iter(train_dataloader)) #What's this, search on net.
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(train_features_batch, train_labels_batch)
print(train_features_batch.shape, train_labels_batch.shape)

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1])
print(random_idx)
random_idx = random_idx.item()
print(random_idx)
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
print(f"Image: {img}, Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
plt.imshow(img.squeeze(), cmap="gray")
plt.axis("Off")
plt.legend()
#plt.show()

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")
print(x)
print(output)

class FashionMNISTModel_v0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_0 = FashionMNISTModel_v0(input_shape=784, hidden_units=10, output_shape=len(class_names)).to(device)
print(model_0)
print(next(model_0.parameters()).device)

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

torch.manual_seed(42)
def eval_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            print(f"Data Loader: {data_loader}, size:{data_loader}")
            x, y = x.to(device), y.to(device)
            print(f"x:{x}, y:{y}")
            y_pred = model(x)
            print(y_pred)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            print(loss, acc)
        
        len_data_loader = len(data_loader)
        loss /= len_data_loader
        acc /= len_data_loader
        print(len_data_loader, loss, acc)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

model_0_result = eval_mode(model=model_0, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
print(model_0_result)

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn: torch.device= device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        train_pred = model(x)
        loss = loss_fn(train_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=train_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        len_data_loader = len(data_loader)
        train_loss /= len_data_loader
        train_acc /= len_data_loader
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.2f}%")

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn: torch.device= device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            test_pred = model(x)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    
        len_data_loader = len(data_loader)
        test_loss /= len_data_loader
        test_acc /= len_data_loader
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc*100:.2f}%")

torch.manual_seed(42)
train_time_start = timer()
epoches = 3

for epoch in tqdm(range(epoches)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, model=model_0, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_step(data_loader=test_dataloader, model=model_0, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)

train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start, end=train_time_end, device=device)
torch.manual_seed(42)
model_0_result = eval_mode(model=model_0, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
print(model_0_result)
#ways to prevent overfitting in machine learning
