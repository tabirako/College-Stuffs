import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = MNISTModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\nModel Architecture:")
print(model)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

print("\nTraining...")
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

torch.save(model.state_dict(), 'mnist_model.pth')
print("\nModel saved to 'mnist_model.pth'")

def run_gui(model, device, transform):
    import tkinter as tk
    from PIL import Image, ImageDraw, ImageOps

    class DigitRecognizer(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("MNIST Digit Recognizer - CNN")
            self.canvas_width = 280
            self.canvas_height = 280
            self.canvas = tk.Canvas(self, width=self.canvas_width,
                                    height=self.canvas_height, bg="white")
            self.canvas.pack()
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
            img = self.image.resize((28, 28))
            img = ImageOps.invert(img)

            img_tensor = transform(img).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()

            self.result_label.config(text=f"Prediction: {pred}")

    app = DigitRecognizer()
    app.mainloop()

if __name__ == "__main__":
    response = input("\nStart GUI for handwritten digit recognition? (y/n): ")
    if response.lower() == 'y':
        run_gui(model, device, transform)
