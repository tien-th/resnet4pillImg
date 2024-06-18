from torchvision import transforms
import dataset # Custom dataset
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import torch.optim as optim
import torch.nn as nn
import numpy as np

def get_random_subset_dataloader(dataset, subset_size, batch_size=32):
    # Randomly sample indices with replacement
    subset_indices = np.random.choice(len(dataset), size=subset_size, replace=True)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
train_dataset = dataset.CustomImageDataset(annotations_file='training/labels.txt', img_dir='training/images', transform=transform)
test_dataset = dataset.CustomImageDataset(annotations_file='public/labels.txt', img_dir='public/images', transform=transform)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    model.train()
    with open('log.txt', 'a') as f:
                f.write(f'\n------------------------------------------------------------------------------------\n')
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # log to txt file 
            with open('log.txt', 'a') as f:
                f.write(f'Epoch {epoch}, Loss: {loss.item()}\n')
            
            
# Parameters
num_models = 1
models = []

for i in range(num_models):
    model = resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
    model.load_state_dict(torch.load('resnet101_model_0.pth'))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Get a unique DataLoader for each model
    
    dataloader = get_random_subset_dataloader(train_dataset, subset_size= int(len(train_dataset) * 0.9), batch_size=32)

    train_model(model, dataloader, criterion, optimizer, num_epochs=80)

    # Save model to disk
    torch.save(model.state_dict(), f'resnet101_model_{i}.pth')
    models.append(model)
def ensemble_predict(models, dataloader):
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = [model(inputs) for model in models]
            avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            all_outputs.append(avg_outputs)
            all_labels.append(labels)

    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_outputs, all_labels

def calculate_accuracy(predictions, labels):
    _, predicted_classes = predictions.max(dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy



test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_predictions, all_labels = ensemble_predict(models, test_loader)
test_accuracy = calculate_accuracy(all_predictions, all_labels)
print(f"Overall Test Accuracy: {test_accuracy * 100:.2f}%")