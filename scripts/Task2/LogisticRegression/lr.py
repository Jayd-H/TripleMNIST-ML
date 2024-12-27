# Jayden Holdsworth 2024
# Task 2: Logistic Regression
# Algorithm to train a Logistic Regression model on the provided triple MNIST dataset

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from PIL import Image
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
from collections import defaultdict
from typing import Tuple, List, Dict
import torch_directml #type: ignore
from sklearn.metrics import f1_score #type: ignore

#* my editor was bugging without the type: ignore, but i couldnt be bothered to fix it

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

class MNISTDataset(Dataset):
    def __init__(self, data_path: str):
        self.images = []
        self.labels = []
        self.image_names = []
        self.label_to_idx = {}
        self.idx = 0
        
        logging.info(f"Loading dataset from {data_path}")
        
        digit_combinations = sorted([d for d in os.listdir(data_path) 
                                  if os.path.isdir(os.path.join(data_path, d)) 
                                  and len(d) == 3 and d.isdigit()])
        
        for digit_combo in digit_combinations:
            if digit_combo not in self.label_to_idx:
                self.label_to_idx[digit_combo] = self.idx
                self.idx += 1
                
            combo_dir = os.path.join(data_path, digit_combo)
            image_files = sorted([f for f in os.listdir(combo_dir) if f.endswith('.png')])
            
            for img_file in image_files:
                img_path = os.path.join(combo_dir, img_file)
                self.images.append(img_path)
                self.labels.append(self.label_to_idx[digit_combo])
                self.image_names.append(img_file)
        
        logging.info(f"Successfully loaded {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.images[idx]).convert('L')
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img_flat = img.reshape(-1)
        img_tensor = torch.FloatTensor(img_flat)
        label = torch.LongTensor([self.labels[idx]])
        return img_tensor, label, self.image_names[idx]

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(7056, 1000)

    def forward(self, x):
        return self.linear(x)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batches = len(loader)
    
    for i, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).squeeze()
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
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
    
    return total_loss / batches, total_correct / total_samples

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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), accuracy, f1, all_preds, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_most_confused_pairs(true_labels: List[int], pred_labels: List[int], output_dir: str, top_n: int = 20):
    logging.info("Creating most confused pairs visualization")
    
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
    plt.savefig(os.path.join(output_dir, 'most_confused_pairs.png'))
    plt.close()

def plot_error_distribution(true_labels: List[int], pred_labels: List[int], output_dir: str):
    logging.info("Creating error distribution visualization")
    
    errors = [abs(true - pred) for true, pred in zip(true_labels, pred_labels)]
    
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Difference Between True and Predicted Labels')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

def main():
    output_dir = os.path.join('scripts', 'Task2', 'LogisticRegression')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch_directml.device()
    logging.info("Initializing training with DirectML (AMD GPU acceleration)")
    
    train_dataset = MNISTDataset('data/train')
    val_dataset = MNISTDataset('data/val')
    test_dataset = MNISTDataset('data/test')
    
    batch_size = 256
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = LogisticRegression().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    logging.info("Starting training")
    
    for epoch in range(1, epochs + 1):
        epoch_start = datetime.now()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_duration = datetime.now() - epoch_start
        
        logging.info(f"Epoch {epoch} completed in {epoch_duration}")
        logging.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")
    
    logging.info("Starting final test evaluation")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    logging.info("Creating visualizations")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir)
    plot_most_confused_pairs(test_labels, test_preds, output_dir)
    plot_error_distribution(test_labels, test_preds, output_dir)
    
    logging.info("Final Test Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()