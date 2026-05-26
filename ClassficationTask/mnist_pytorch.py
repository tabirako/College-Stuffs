# %%
## https://qiita.com/automation2025/items/12c19e31c85f9ea1e9ae#chapter-1-pytorch%E3%81%AE%E6%A6%82%E8%A6%81

import torch

# Tensorの作成
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor x:", x)


# %%
# Tensorの加算
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = a + b
print("Tensor a:", a)
print("Tensor b:", b)
print("Sum c:", c)

# %%
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward()
print("Gradient of y with respect to x:", x.grad)

# %%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 1)  # 2入力、1出力の全結合層

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
print("Simple Neural Network Model:", model)

# %%
from torch.utils.data import DataLoader, TensorDataset

# ダミーデータ
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
labels = torch.tensor([0, 1])

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print("Batch:", batch)

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5):
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)  # フォワードパス
        loss = criterion(outputs, targets.float().unsqueeze(1))  # 損失の計算
        loss.backward()  # バックプロパゲーション
        optimizer.step()  # パラメータの更新
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# %%
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        print("Model outputs:", outputs)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets.float().unsqueeze(1))
    loss.backward()
    optimizer.step()

# %%
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

custom_dataset = CustomDataset(data, labels)
print("Custom Dataset Length:", len(custom_dataset))

# %%
from torchvision import models

pretrained_model = models.resnet18(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False  # パラメータを固定

pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 2)  # 新しい出力層
print("Modified ResNet Model:", pretrained_model)

# %%
# モデルの保存
torch.save(model.state_dict(), 'model.pth')
print("Model saved to 'model.pth'.")

# モデルの読み込み
model.load_state_dict(torch.load('model.pth'))
print("Model loaded from 'model.pth'.")

# %%
# 学習率の変更
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
print("Optimizer learning rate set to 0.001.")

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

cnn_model = SimpleCNN()
print("Simple CNN Model:", cnn_model)

# %%
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

rnn_model = SimpleRNN(input_size=10, hidden_size=20, output_size=1)
print("Simple RNN Model:", rnn_model)

# %%
# トークン化と埋め込み
"""
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
tokens = tokenizer("Hello PyTorch")
print("Tokens:", tokens)



ModuleNotFoundError                       Traceback (most recent call last)
Cell In[23], line 2
      1 # トークン化と埋め込み
----> 2 from torchtext.data.utils import get_tokenizer
      3 tokenizer = get_tokenizer('basic_english')
      4 tokens = tokenizer("Hello PyTorch")

ModuleNotFoundError: No module named 'torchtext'

"""

# %%
# デバッグ用のprintステートメント
print(f"Input: {inputs}, Output: {outputs}")

# %%
# Adamオプティマイザーの使用
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Using Adam optimizer with learning rate 0.001.")

# %%
# モデルのトレーニングと評価を分離する
def train(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# %%
# torch.jitを使用したモデルのスクリプト化
scripted_model = torch.jit.script(model)
print("Scripted Model:", scripted_model)

# %%
from torchvision import datasets, transforms

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# モデルの定義
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# モデルのトレーニング
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# %%
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Recognizer")
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self, width=self.canvas_width,
                                height=self.canvas_height, bg="white")
        self.canvas.pack()
        # keep a PIL image to record the strokes
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(side="left")
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")
        self.result_label = tk.Label(self, text="Draw a digit and click predict")
        self.result_label.pack(side="left")

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=21, fill="black",
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=0, width=21)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height],
                            fill=255)
        self.result_label.config(text="Draw a digit and click predict")

    def predict(self):
        # resize to 28x28, invert colours (MNIST is white-on‑black)
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        plt.imshow(img, cmap="gray")
        plt.title("Processed Image for Prediction")
        arr = np.array(img, dtype=np.float32).reshape(1, -1)
        arr_pca = pca.transform(arr)           # existing PCA instance
        pred = linear_svm_clf.predict(arr_pca) # existing trained classifier
        self.result_label.config(text=f"Prediction: {pred[0]}")

# run the GUI
if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()


