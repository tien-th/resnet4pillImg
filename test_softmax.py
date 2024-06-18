from torchvision import transforms
import dataset # Custom dataset
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold


# def get_random_subset_dataloader(dataset, subset_size, batch_size=32):
#     # Randomly sample indices with replacement
    
#     subset_indices = np.random.choice(len(dataset), size=subset_size, replace=True)
#     subset = torch.utils.data.Subset(dataset, subset_indices)
#     dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
#     return dataloader

batch_size = 64 
# log_file = 'log_5_fold.txt'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
test_dataset = dataset.CustomImageDataset(annotations_file='public/labels.txt', img_dir='public/images', transform=transform)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

def infer_with_model(model, dataloader):
    model.to(device)
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            all_outputs.append(softmax_outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    model.to('cpu')
    
    return all_outputs, all_labels

def ensemble_predict(models, dataloader):
    all_model_outputs = []

    # Infer with each model and collect all softmax outputs
    for model in models:
        model_outputs, all_labels = infer_with_model(model, dataloader)
        all_model_outputs.append(model_outputs)

    # Stack all model outputs and average them
    avg_outputs = torch.mean(torch.stack(all_model_outputs), dim=0)

    return avg_outputs, all_labels

def calculate_accuracy(predictions, labels):
    _, predicted_classes = predictions.max(dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
# Example usage during testing



models = []
for i in range(5): 
    model = resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
    model.load_state_dict(torch.load(f'resnet101_model_{i}.pth'))
    models.append(model)
    
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
all_predictions, all_labels = ensemble_predict(models, test_loader)
test_accuracy = calculate_accuracy(all_predictions, all_labels)
print(f"Overall Test Accuracy: {test_accuracy * 100:.2f}%")


# def train_model(model, dataloader, criterion, optimizer, num_epochs=30, model_fold=0):
#     model.train()
  
#     for epoch in range(num_epochs):
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # log to txt file 
           
        
#         val_acc = test([model])
            
# Parameters
# num_models = 5
# models = []

# for i in range(num_models):
#     model = resnet101(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
#     # model.load_state_dict(torch.load('resnet101_model_0.pth'))
#     model.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     criterion = nn.CrossEntropyLoss()

#     # Get a unique DataLoader for each model
    
#     train_dataset = dataset.CustomImageDataset(annotations_file=f'training/fold{i}_train.txt', img_dir='training/images', transform=transform)
#     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     print(f"Training model {i}") 
#     train_model(model, dataloader, criterion, optimizer, num_epochs=40, model_fold=i)

#     # Save model to disk
#     torch.save(model.state_dict(), f'resnet101_model_{i}.pth')
#     models.append(model)

