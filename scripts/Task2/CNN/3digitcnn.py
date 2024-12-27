# Jayden Holdsworth 2024
# Task 2: CNN
# Algorithm to train a CNN model on the provided triple MNIST dataset
# Treats each image as three digits with 1000 different outputs

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from PIL import Image
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from typing import Tuple, List
import torch_directml #type: ignore
import seaborn as sns #type: ignore
from datetime import datetime
from collections import defaultdict
import warnings

#* my editor was bugging without the type: ignore, but i couldnt be bothered to fix it

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='sklearn.metrics._classification')

#* funny warnings that i dont want to see lol

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def setup_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join('scripts', 'Task2', 'CNN', 'results', timestamp)
    vis_dir = os.path.join(base_dir, 'visualisations')
    feature_map_dir = os.path.join(vis_dir, 'FeatureMaps')
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(feature_map_dir, exist_ok=True)
    
    return base_dir, vis_dir, feature_map_dir

def create_run_log_header(filepath: str, model, batch_size: int, epochs: int, learning_rate: float):
    with open(filepath, 'w') as f:
        f.write("=== CNN Training Run Log ===\n\n")
        f.write("This log file contains the following information:\n")
        f.write("1. Configuration parameters\n")
        f.write("2. Per-epoch training and validation metrics\n")
        f.write("3. Final evaluation metrics\n\n")
        f.write("Metrics explanation:\n")
        f.write("- Loss: Cross-entropy loss between predicted and actual labels\n")
        f.write("- Accuracy: Percentage of correctly classified images\n")
        f.write("- F1 Score: Harmonic mean of precision and recall\n\n")
        f.write("=== Configuration ===\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write("\nModel Architecture:\n")
        f.write(str(model))
        f.write("\n\n=== Training Log ===\n")

class TripleMNISTDataset(Dataset):
    def __init__(self, data_path: str):
        self.images = []
        self.labels = []
        self.image_names = []
        
        logging.info(f"Loading dataset from {data_path}")
        
        digit_combinations = sorted([d for d in os.listdir(data_path) 
                                  if os.path.isdir(os.path.join(data_path, d)) 
                                  and len(d) == 3 and d.isdigit()])
        
        for digit_combo in digit_combinations:
            combo_dir = os.path.join(data_path, digit_combo)
            image_files = sorted([f for f in os.listdir(combo_dir) if f.endswith('.png')])
            
            for img_file in image_files:
                img_path = os.path.join(combo_dir, img_file)
                self.images.append(img_path)
                self.labels.append(int(digit_combo))
                self.image_names.append(img_file)
        
        logging.info(f"Successfully loaded {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.images[idx]).convert('L')
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])
        return img_tensor, label, self.image_names[idx]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        
        feature_size = 64 * (84 // 4) * (84 // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1000)
        )

    def forward(self, x):
        features = self.features(x)
        x = features.view(features.size(0), -1)
        x = self.classifier(x)
        return x, features

def plot_feature_maps(model, sample_image, feature_map_dir, epoch):
    if epoch == 1 or epoch % 5 == 0:
        logging.info(f"Creating feature map visualisation for epoch: {epoch}")
        model.eval()
        with torch.no_grad():
            _, feature_maps = model(sample_image.unsqueeze(0))
        
        feature_map = feature_maps[0, 0].cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature Map at Epoch {epoch}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(feature_map_dir, f'feature_map_epoch_{epoch}.png'))
        plt.close()

def plot_most_confused_pairs(true_labels: List[int], pred_labels: List[int], vis_dir: str, top_n: int = 20):
    logging.info("Creating most confused pairs visualisation")
    
    confusion_dict = defaultdict(int)
    for true, pred in zip(true_labels, pred_labels):
        if true != pred:
            confusion_dict[(true, pred)] += 1
    
    most_confused = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    pairs = [f"{pair[0][0]}->{pair[0][1]}" for pair in most_confused]
    frequencies = [pair[1] for pair in most_confused]
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x=frequencies, y=pairs)
    plt.title(f'Top {top_n} Most Confused Pairs')
    plt.xlabel('Frequency')
    plt.ylabel('True->Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'most_confused_pairs.png'))
    plt.close()

def plot_error_distribution(true_labels: List[int], pred_labels: List[int], vis_dir: str):
    logging.info("Creating error distribution visualisation")
    
    errors = [abs(true - pred) for true, pred in zip(true_labels, pred_labels)]
    
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Difference Between True and Predicted Labels')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
    plt.close()

def plot_training_history(train_losses, train_accs, val_losses, val_accs, vis_dir):
    logging.info("Creating training history visualisation")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, sample_image=None, feature_map_dir=None, log_file=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batches = len(loader)
    epoch_start = datetime.now()
    
    for i, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).squeeze()
        
        optimizer.zero_grad(set_to_none=True)
        outputs, _ = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        if (i + 1) % max(1, batches // 10) == 0:
            accuracy = total_correct / total_samples
            avg_loss = total_loss / (i + 1)
            logging.info(f'Epoch {epoch}: Batch {i+1}/{batches} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    if sample_image is not None and feature_map_dir is not None:
        plot_feature_maps(model, sample_image, feature_map_dir, epoch)
    
    epoch_duration = datetime.now() - epoch_start
    epoch_loss = total_loss / batches
    epoch_acc = total_correct / total_samples
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nEpoch {epoch} (Duration: {epoch_duration}):\n")
            f.write(f"Training Loss: {epoch_loss:.4f}\n")
            f.write(f"Training Accuracy: {epoch_acc:.4f}\n")
    
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    logging.info("Starting evaluation")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze()
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    f1_score = report['weighted avg']['f1-score']
    
    return total_loss / len(loader), accuracy, all_preds, all_labels, f1_score

def main():
    base_dir, vis_dir, feature_map_dir = setup_run_directory()
    log_file = os.path.join(base_dir, 'run_log.txt')
    
    device = torch_directml.device()
    logging.info("Initializing training with DirectML (AMD GPU acceleration)")
    
    batch_size = 256
    epochs = 10
    learning_rate = 0.001
    
    train_dataset = TripleMNISTDataset('data/train')
    val_dataset = TripleMNISTDataset('data/val')
    test_dataset = TripleMNISTDataset('data/test')
    
    sample_image, _, _ = train_dataset[0]
    sample_image = sample_image.to(device)
    
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    create_run_log_header(log_file, model, batch_size, epochs, learning_rate)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    training_start_time = datetime.now()
    logging.info("Starting training")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            sample_image=sample_image, feature_map_dir=feature_map_dir,
            log_file=log_file
        )
        val_loss, val_acc, _, _, val_f1 = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        with open(log_file, 'a') as f:
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n")
            f.write(f"Validation F1 Score: {val_f1:.4f}\n")
        
        logging.info(f"Epoch {epoch}:")
        logging.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    training_duration = datetime.now() - training_start_time
    
    logging.info("Starting final test evaluation")
    test_loss, test_acc, test_preds, test_labels, test_f1 = evaluate(model, test_loader, criterion, device)
    
    logging.info("Creating visualisations")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, vis_dir)
    plot_most_confused_pairs(test_labels, test_preds, vis_dir)
    plot_error_distribution(test_labels, test_preds, vis_dir)
    
    with open(log_file, 'a') as f:
        f.write("\n=== Final Results ===\n")
        f.write(f"Total Training Time: {training_duration}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    
    logging.info("Final Test Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    logging.info(f"Results saved to {base_dir}")
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()