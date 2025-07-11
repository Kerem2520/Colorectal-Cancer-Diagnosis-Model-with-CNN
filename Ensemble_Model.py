import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

siniflar = [
    "ADI", "BACK", "DEB", "LYM",
    "MUC", "MUS", "NORM", "STR","TUM"
]

veri_yolu = "C:/Users/musta/Desktop/CRC-VAL-HE-7K"

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 16

class HistopathologyDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for i, sinif in enumerate(classes):
            sinif_dizin = os.path.join(root_dir, sinif)
            for dosya in os.listdir(sinif_dizin):
                self.image_paths.append(os.path.join(sinif_dizin, dosya))
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

dataset = HistopathologyDataset(veri_yolu, siniflar, transform)

train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.25, random_state=42, stratify=dataset.labels)

train_set = torch.utils.data.Subset(dataset, train_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

class LateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionModel, self).__init__()

        self.resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet = self.resnet.to(device)

        self.efficientnet = models.efficientnet_b1(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Identity()
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        self.efficientnet = self.efficientnet.to(device)

        self.fc = nn.Linear(2048 + 1280, num_classes)

    def forward(self, x):
        with torch.no_grad():
            resnet_features = self.resnet(x)
            efficientnet_features = self.efficientnet(x)

        fused_features = torch.cat((resnet_features, efficientnet_features), dim=1)
        return self.fc(fused_features)

model = LateFusionModel(num_classes=len(siniflar)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_f1 = f1_score(all_labels, all_predictions, average='weighted')

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_all_labels.extend(labels.cpu().numpy())
            test_all_predictions.extend(predicted.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    test_f1 = f1_score(test_all_labels, test_all_predictions, average='weighted')

    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Train F1 Score: {train_f1:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}")


torch.save(model.state_dict(), "late_fusion_model.pth")
print("Model başarıyla kaydedildi: late_fusion_model.pth")


































