import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x
        
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def show_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    for i in range(5):
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Prediction: {preds[i].item()}")
        plt.axis("off")
        plt.show()

def main():
    # Where the model will be saved
    model_path = "digit_cnn.pth"

    print("Loading data...")
    train_loader, test_loader = load_data()

    # Initialize the model with the type of algorithm we will use.
    print("Building model...")
    model = DigitCNN()

    # Ask user if they want to use a saved model.
    use_saved = input("Load saved model? (y/n): ").strip().lower()

    if use_saved == "y":
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Loaded saved model.")

            print("Evaluating...")
            evaluate_model(model, test_loader)

            print("Showing sample predictions...")
            show_predictions(model, test_loader)
            
            return
        else:
            print("No saved model found, training new model...")

    print("Training...")
    train_model(model, train_loader, epochs=5)

    torch.save(model.state_dict(), model_path)

    print("Evaluating...")
    evaluate_model(model, test_loader)

    print("Showing sample predictions...")
    show_predictions(model, test_loader)

if __name__ == "__main__":
    main()