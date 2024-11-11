import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms, models # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from datetime import datetime
import pickle
import random


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CNN Model Architecture
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassCNN(nn.Module):
    def __init__(self, first_neuron, number_of_class):
        super(MultiClassCNN, self).__init__()
        
        self.first_out_channels = first_neuron
        
        # Convolutional layers with dropout and batch normalization
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=self.first_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            # Second Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels, out_channels=self.first_out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*2),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            # Third Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels*2, out_channels=self.first_out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*4),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),

            # Fourth Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels*4, out_channels=self.first_out_channels*8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*8),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            
            # Global Average Pooling (GAP)
            nn.AdaptiveAvgPool2d(1),  # GAP
        )
        
        # Flatten the output of the GAP layer
        self.flatten_size = self.first_out_channels * 8
        
        # Fully connected classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Dropout(0.4),
            nn.Linear(128, number_of_class),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output of GAP layer
        x = self.classifier(x)
        return x

# Initialize the model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

# Data augmentation transforms
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).long()  # Convert labels to long for cross-entropy
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1) # Find maximum value
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, scheduler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device).long()  # Convert labels to long for cross-entropy
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    if scheduler:
        scheduler.step(epoch_loss)  # Using validation loss here
        
    return epoch_loss, epoch_acc, all_preds, all_labels

import matplotlib.pyplot as plt
from collections import defaultdict

class ModelTracker:
    def __init__(self, model_params):
        self.history = defaultdict(list)
        self.model_params = model_params
    
    def update(self, metrics):
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def get_best_validation_metrics(self):
        val_accs = self.history['val_acc']
        best_acc = max(val_accs)
        best_epoch = val_accs.index(best_acc) + 1  # +1 because epochs start from 1
        return best_acc, best_epoch
    
    def plot_metrics(self):
        """Plot training and validation metrics with parameters."""
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 10))
        
        # Create grid spec for custom layout
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        
        # Loss subplot
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', marker='o')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', marker='o')
        ax1.set_title('Model Loss', fontsize=12, pad=10)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Accuracy subplot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', marker='o')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', marker='o')
        ax2.set_title('Model Accuracy', fontsize=12, pad=10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Parameters text box
        params_ax = fig.add_subplot(gs[:, 1])
        params_ax.axis('off')
        
        # Create parameter text
        params_text = "Model Parameters:\n" + "="*30 + "\n"
        for param, value in self.model_params.items():
            params_text += f"{param:<20}: {value}\n"
        
        # Add overfitting analysis
        overfitting_analysis = self.check_overfitting()
        params_text += "\nOverfitting Analysis:\n" + "="*30 + "\n"
        params_text += overfitting_analysis
        
        # Get best validation metrics
        best_val_acc, best_epoch = self.get_best_validation_metrics()
        
        # Add final metrics
        params_text += "\n\nFinal Metrics:\n" + "="*30 + "\n"
        params_text += f"Training Loss: {self.history['train_loss'][-1]:.4f}\n"
        params_text += f"Validation Loss: {self.history['val_loss'][-1]:.4f}\n"
        params_text += f"Training Accuracy: {self.history['train_acc'][-1]:.4f}%\n"
        params_text += f"Validation Accuracy: {self.history['val_acc'][-1]:.4f}%\n"
        params_text += f"\nBest Validation Accuracy: {best_val_acc:.4f}%\n"
        params_text += f"Best Validation Epoch: {best_epoch}"
        
        # Add text to figure
        params_ax.text(0, 0.95, params_text, transform=params_ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      family='monospace', bbox=dict(facecolor='white', 
                                                  alpha=0.8,
                                                  edgecolor='gray',
                                                  boxstyle='round,pad=1'))
        
        # Add timestamp and title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f"Training Results - {timestamp}", fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def check_overfitting(self, threshold=5):
        """Check for signs of overfitting"""
        if len(self.history['train_acc']) < 2:
            return "Not enough epochs to determine overfitting"
        
        train_acc = self.history['train_acc'][-1]
        val_acc = self.history['val_acc'][-1]
        acc_diff = train_acc - val_acc
        
        val_loss_trend = self.history['val_loss'][-3:]  # Last 3 epochs
        
        analysis = []
        if acc_diff > threshold:
            analysis.append(f"Warning: Training accuracy exceeds\nvalidation accuracy by {acc_diff:.2f}%")
        
        if len(val_loss_trend) == 3 and all(val_loss_trend[i] > val_loss_trend[i-1] for i in range(1, len(val_loss_trend))):
            analysis.append("Warning: Validation loss is\nconsistently increasing")
            
        return "\n".join(analysis) if analysis else "No clear signs of overfitting detected"

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    # Define model parameters
    model_params = {
        "Batch Size": train_loader.batch_size,
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Epochs": num_epochs,
        "Criterion": criterion.__class__.__name__,
        "Optimizer": optimizer.__class__.__name__,
        "First Out Channels": model.first_out_channels,
        "Model Architecture": "CatDogCNN",
        "Device": device,
        "Total Parameters": sum(p.numel() for p in model.parameters()),
        "Trainable Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    tracker = ModelTracker(model_params)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,scheduler)
        
        # Validation
        val_loss, val_acc, val_pred, val_labels = validate(model, val_loader, criterion, device, scheduler)
        
        # Update metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        tracker.update(metrics)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
    
    # Plot final results
    fig = tracker.plot_metrics()

    # Save the figure
    output_dir = "Final_Image"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(output_dir, f"Training Results - {timestamp}.jpg")
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return tracker

def set_reproducibility_seeds(seed=42):
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_deterministic_loaders(train_dataset, val_dataset, batch_size, num_workers=0):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(42),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    # 1. Set all seeds
    set_reproducibility_seeds(42)
    
    # 2. Set deterministic behavior for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
    # 3. GPU Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Using device: {device}")
    
    # 4. Load Dataset (Please Download from google drive -> link is in the ReadME file)
    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # Assuming the extracted files are in the directory 'cifar-10-batches-py'
    data_path = 'cifar-10-batches-py'
    batch_files = [f'data_batch_{i}' for i in range(1, 6)]
    test_file = 'test_batch'

    # Load all training batches
    train_data = []
    train_labels = []

    for batch_file in batch_files:
        batch = unpickle(os.path.join(data_path, batch_file))
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    # Convert lists to single arrays
    import numpy as np

    train_data = np.vstack(train_data)  # Stack into one numpy array
    train_labels = np.array(train_labels)

    # Load the test batch
    test_batch = unpickle(os.path.join(data_path, test_file))
    test_data = np.array(test_batch[b'data'])
    test_labels = np.array(test_batch[b'labels'])

    reshape_train_data = train_data.reshape(50000, 3, 32, 32)
    reshape_val_data = test_data.reshape(10000, 3, 32, 32)

    pixels_list = reshape_train_data.tolist()
    pixels_list_val = reshape_val_data.tolist()

    train = pd.DataFrame({
                'pixels': pixels_list,
                'label': train_labels
            })

    test = pd.DataFrame({
                'pixels': pixels_list_val,
                'label': test_labels
            })
    
    train['pixels'] = train['pixels'].apply(lambda x: np.array(x) / 255.0)
    test['pixels'] = test['pixels'].apply(lambda x: np.array(x) / 255.0)

    print("2. Dataset Successfully Imported")
    
    pixel_data = np.array(train['pixels'].tolist())
    labels = np.array(train['label'].tolist())
    X = torch.tensor(pixel_data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    test_pixel_data = np.array(test['pixels'].tolist())
    test_labels = np.array(test['label'].tolist())
    val_X = torch.tensor(test_pixel_data, dtype=torch.float32)
    val_y = torch.tensor(test_labels, dtype=torch.float32)
    
    train_transforms = get_train_transforms()

    augmented_images = []
    for img in X:
        img = transforms.ToPILImage()(img)           
        augmented_img = train_transforms(img)
        augmented_images.append(augmented_img)

    val_augmented_images = []
    for img in val_X:
        img = transforms.ToPILImage()(img)
        val_augmented_img = train_transforms(img)
        val_augmented_images.append(val_augmented_img)

    augmented_images = torch.stack(augmented_images)
    val_augmented_images = torch.stack(val_augmented_images)

    combined_images = torch.cat([X, augmented_images], dim=0)
    combined_labels = torch.cat([y, y], dim=0)  # Duplicate labels as well

    val_combined_images = torch.cat([val_X, val_augmented_images], dim=0)
    val_combined_labels = torch.cat([val_y, val_y], dim=0)  # Duplicate labels as well

    train_dataset = TensorDataset(combined_images, combined_labels)
    val_dataset = TensorDataset(val_combined_images, val_combined_labels)

    print("3. Tensor Convertion Completed")
    print()
    
    # 5. Create deterministic dataloaders
    BATCH_SIZE = int(input("Enter Batch Size (recommended 128): "))
    train_loader, val_loader = create_deterministic_loaders(
        train_dataset,
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0  # Set to 0 for complete reproducibility
    )
    
    # 6. Model initialization with fixed seed
    torch.manual_seed(42)
    neuron = int(input("Enter number of second neurons (recommended 128): "))
    model = MultiClassCNN(neuron, 10).to(device)
    model.apply(init_weights)
    
    # 7. Initialize optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    learning_rate = float(input("Enter learning rate value (recommended 0.001): "))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=  0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 8. Train model
    NUM_EPOCHS = int(input('Enter number of epochs (recommended 18): '))
    print()
    print('Parameter used:')
    print(f'Learning Rate = {learning_rate}')
    print(f'Batch Size = {BATCH_SIZE}')
    print(f'Second Neurons = {neuron}')
    print(f'Number of Epochs: {NUM_EPOCHS}')
    print()
    print("-------------Training Begin-------------")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        scheduler=scheduler
    )