from torchvision.models import SwinTransformer, swin_t, Swin_T_Weights
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_model() -> SwinTransformer:
    weights = Swin_T_Weights.DEFAULT
    return swin_t(weights = weights)

def train_model(model: SwinTransformer, dataset_name: str, num_epochs: int = 10) -> SwinTransformer:
    num_classes = 200
    model.head = nn.Linear(model.head.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    dataset_path = "../../data/" + dataset_name
    train_dataset = datasets.ImageFolder(root = dataset_path, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    return model

model = load_model()
model = train_model(model, "cub-200-2011/CUB_200_2011/images")
model.eval()

