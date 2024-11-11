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
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Model Architecture
class CatDogCNN(nn.Module):
    def __init__(self, first_neuron):
        super(CatDogCNN, self).__init__()

        self.first_out_channels = first_neuron
        
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=self.first_out_channels, kernel_size=3, padding=1),  # dimension [batch_size, out_channel, 128, `128`] -> padding = 1
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel, 64, 64]
            
            # Second Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels, out_channels=self.first_out_channels*2, kernel_size=3, padding=1), # dimension [batch_size, out_channel, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*2),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel*2, 32, 32]
            
            # Third Convolutional Block
            nn.Conv2d(in_channels=self.first_out_channels*2, out_channels=self.first_out_channels*4, kernel_size=3, padding=1), # dimension [batch_size, out_channel, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(self.first_out_channels*4),
            nn.MaxPool2d(2), # dimension [batch_size, out_channel*4, 16, 16]
            
        )
        
        # Calculate size of flattened features after the convolutional layers
        self.flatten_size = self.first_out_channels * 4 * (128 // (2**3)) * (128 // (2**3)) # out_channel * (128 // 2^amount_of_max_pool) * (128 // 2^amount_of_max_pool)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flatten_size, 256), # dimension [batch_size, 256]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), # dimension [batch_size, 128]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1), # dimension [batch_size, 1]
            nn.Sigmoid() # To get value from 0 to 1
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

# Training Functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

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
        params_text += f"Training Accuracy: {self.history['train_acc'][-1]:.5f}%\n"
        params_text += f"Validation Accuracy: {self.history['val_acc'][-1]:.5f}%\n"
        params_text += f"\nBest Validation Accuracy: {best_val_acc:.5f}%\n"
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_pred, val_labels = validate(model, val_loader, criterion, device)
        
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
    fig.savefig(file_path, dpi=300, bbox_inches='tight') # Save the result in jpg is optional
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

def create_deterministic_loaders(X, y, val_X, val_y, batch_size, num_workers=0):
    # Create datasets
    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(val_X, val_y)
    
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

def predict_test_images(model, test_folder, device):
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store results
    image_ids = []
    predictions = []
    
    # Process each image
    print("Processing test images...")
    for filename in tqdm(sorted(os.listdir(test_folder), key=lambda x: int(x.split('.')[0]))):
        if filename.endswith('.jpg'):
            # Extract image ID
            image_id = int(filename.split('.')[0])
            
            # Load and preprocess image
            image_path = os.path.join(test_folder, filename)
            image = Image.open(image_path).resize((128,128)).convert('RGB')
            image_array = np.array(image)
            
            # Convert to Tensor
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Make prediction
            with torch.no_grad():
                inputs = image_tensor.to(device)
                output = model(inputs)
                pred = (output.squeeze() > 0.5).float()
            
            # Store results
            image_ids.append(image_id)
            predictions.append(int(pred))  # Convert to int for cleaner CSV
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'id': image_ids,
        'label': predictions
    })
    
    # Sort by ID
    results_df = results_df.sort_values('id')
    
    # Save to CSV
    csv_path = 'submission.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")
    
    return results_df

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
    train = pd.read_pickle('train_128.pkl')
    val = pd.read_pickle('test_128.pkl')
    print("2. Dataset Successfully Imported")
    
    # Convert to input and output
    pixel_data = np.array(train['pixels'].tolist())
    labels = np.array(train['label'].tolist())
    val_pixel_data = np.array(val['pixels'].tolist())
    val_labels = np.array(val['label'].tolist())
    
    # Convert to tensors with fixed seeds
    X = torch.tensor(pixel_data, dtype=torch.float32)
    X = X.permute(0, 3, 1, 2)
    y = torch.tensor(labels, dtype=torch.float32)
    
    val_X = torch.tensor(val_pixel_data, dtype=torch.float32)
    val_X = val_X.permute(0, 3, 1, 2)
    val_y = torch.tensor(val_labels, dtype=torch.float32)
    print("3. Tensor Convertion Completed")
    print()
    
    # 5. Create deterministic dataloaders
    BATCH_SIZE = int(input("Enter Batch Size (recommended 64): "))
    train_loader, val_loader = create_deterministic_loaders(
        X, y, val_X, val_y, 
        batch_size=BATCH_SIZE,
        num_workers=0  # Set to 0 for complete reproducibility
    )
    
    # 6. Model initialization with fixed seed
    torch.manual_seed(42)
    neuron = int(input("Enter number of second neurons (recommended 128): "))
    model = CatDogCNN(neuron).to(device)
    
    # 7. Initialize optimizer and criterion
    criterion = nn.BCELoss()
    learning_rate = float(input("Enter learning rate value (recommended 0.005): "))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 8. Train model
    NUM_EPOCHS = int(input('Enter number of epochs (recommended 28): '))
    print()
    print('Parameter used:')
    print(f'Learning Rate = {learning_rate}')
    print(f'Batch Size = {BATCH_SIZE}')
    print(f'Second Neurons = {neuron}')
    print(f'Number of Epochs: {NUM_EPOCHS}')
    print()
    print("-------------Training Begin-------------")

    tracker = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device
    )

    # # 9. Save Model (Optional)
    # save_path = 'cnnmodel'

    # torch.save(model.state_dict(), save_path)
    # print(f"Model state saved to: {save_path}")

    # # 10. Load the Model
    # load_path = 'cnnmodel'

    # state_dict = torch.load(load_path)
    # model.load_state_dict(state_dict)
    # print(f"Model state loaded from: {load_path}")

    # # 11. Submission Evaluation
    # DATASET_FOLDER = 'datasets/test'
    # results = predict_test_images(model, DATASET_FOLDER, device)

    #  # Display first few predictions
    # print("\nFirst few predictions:")
    # print(results.head())
    # print(f"\nTotal images processed: {len(results)}")
    
    # # Display label distribution
    # print("\nLabel distribution:")
    # print(results['label'].value_counts())
