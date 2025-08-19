import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# ======================
# 1. Dataset Definition
# ======================
class AffectNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file (annotations.csv)
            root_dir (str): Directory with images
            transform (callable, optional): Optional transform
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])  # Emotion label (0–7)

        if self.transform:
            image = self.transform(image)

        return image, label

# ======================
# 2. Model Definition
# ======================
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 8 emotions

    def forward(self, x):
        return self.resnet(x)

# (Later you can extend this with Vid2Seq for temporal modeling)

# ======================
# 3. Training Function
# ======================
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")

    print("Training complete!")
    return model

# ======================
# 4. Main Execution
# ======================
if __name__ == "__main__":
    # Paths (adjust these!)
    train_csv = "data/annotations.csv"
    train_dir = "data/train"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    dataset = AffectNetDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionResNet(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Save model
    torch.save(model.state_dict(), "emotion_resnet50.pth")
    print("Model saved as emotion_resnet50.pth")
