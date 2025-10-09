import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from data_setup import train_loader, test_loader, device

def train_model():
    # ✅ Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # ✅ Replace the final layer for 4 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    # ✅ Move to GPU and wrap with DataParallel (simulated multi-GPU training)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # ✅ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training loop (1–2 epochs for demo)
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(train_loader))

    print("✅ Training complete!")

    # ✅ Save model
    torch.save(model.state_dict(), "recycle_resnet18.pth")
    print("Model saved as recycle_resnet18.pth")

if __name__ == "__main__":
    train_model()
