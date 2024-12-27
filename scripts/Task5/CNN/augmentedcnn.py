# Task 5: Augmented CNN
# Trains 2 models, one with only real data and one with a combination of real and synthetic data
# Architecture is the take as Task 4

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import Dataset, DataLoader, ConcatDataset #type: ignore
from torchvision import transforms #type: ignore
from PIL import Image
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch_directml #type: ignore
import seaborn as sns #type: ignore
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='sklearn.metrics._classification')

#* editor was bugging without the type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def setup_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join('scripts', 'Task5', 'CNN', 'results', timestamp)
    
    os.makedirs(base_dir, exist_ok=True)
    root_vis_dir = os.path.join(base_dir, 'visualisations')
    os.makedirs(root_vis_dir, exist_ok=True)
    
    model_dirs = ['real_only', 'combined']
    paths = {
        'base': base_dir,
        'root_vis': root_vis_dir
    }
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        vis_path = os.path.join(model_path, 'visualisations')
        os.makedirs(vis_path, exist_ok=True)
        paths[model_dir] = {
            'base': model_path,
            'vis': vis_path
        }
    
    return paths

def create_run_log_header(filepath: str, model, batch_size: int, epochs: int, learning_rate: float):
    with open(filepath, 'w') as f:
        f.write("=== CNN Training Run Log ===\n\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n\n")
        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write("\n\n=== Training Logs ===\n")

class TripleMNISTDataset(Dataset):
    def __init__(self, data_path: str, train: bool = False, is_synthetic: bool = False):
        self.images = []
        self.labels = []
        self.image_names = []
        self.train = train
        self.is_synthetic = is_synthetic
        
        if self.train and not self.is_synthetic:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),
                                      scale=(0.9, 1.1), shear=5),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        
        digit_combinations = sorted([d for d in os.listdir(data_path) 
                                  if os.path.isdir(os.path.join(data_path, d)) 
                                  and len(d) == 3 and d.isdigit()])
        
        for digit_combo in digit_combinations:
            combo_dir = os.path.join(data_path, digit_combo)
            image_files = sorted([f for f in os.listdir(combo_dir) if f.endswith('.png')])
            
            for img_file in image_files:
                img_path = os.path.join(combo_dir, img_file)
                self.images.append(img_path)
                self.labels.append([int(d) for d in digit_combo])
                self.image_names.append(img_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        
        if self.train and not self.is_synthetic:
            img = self.transform(img)
        
        img = np.array(img).astype(np.float32) / 255.0
        digits = torch.FloatTensor(img).unsqueeze(0)
        label = torch.LongTensor(self.labels[idx])
        
        return digits, label, self.image_names[idx]

class CNN(nn.Module):
    def __init__(self, dropout_rate=0.15):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )
        
        feature_size = 96 * (84 // 8) * (28 // 8)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        width = x.size(3)
        split_width = width // 3
        
        digit1 = x[:, :, :, :split_width]
        digit2 = x[:, :, :, split_width:2*split_width]
        digit3 = x[:, :, :, 2*split_width:]
        
        d1 = self.features(digit1)
        d2 = self.features(digit2)
        d3 = self.features(digit3)
        
        d1 = d1.view(batch_size, -1)
        d2 = d2.view(batch_size, -1)
        d3 = d3.view(batch_size, -1)
        
        out1 = self.classifier(d1)
        out2 = self.classifier(d2)
        out3 = self.classifier(d3)
        
        return torch.stack([out1, out2, out3], dim=1)

def evaluate_with_predictions(model, loader, criterion, device):
   model.eval()
   total_loss = 0
   total_correct = 0
   total_samples = 0
   digit_preds = [[], [], []]
   digit_labels = [[], [], []]
   
   with torch.no_grad():
       for images, labels, _ in loader:
           images = images.to(device)
           labels = labels.to(device)
           
           outputs = model(images)
           loss = criterion(outputs.view(-1, 10), labels.view(-1))
           
           _, predicted = outputs.max(2)
           correct = (predicted == labels).all(dim=1).sum().item()
           total_correct += correct
           total_samples += labels.size(0)
           total_loss += loss.item()
           
           for i in range(3):
               digit_preds[i].extend(predicted[:, i].cpu().numpy())
               digit_labels[i].extend(labels[:, i].cpu().numpy())
   
   accuracy = total_correct / total_samples
   digit_cms = [confusion_matrix(digit_labels[i], digit_preds[i], labels=range(10)) 
               for i in range(3)]
   
   f1_scores = []
   for i in range(3):
       cm = digit_cms[i]
       precisions = np.diag(cm) / (cm.sum(axis=0) + 1e-15)
       recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-15)
       f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
       f1_scores.append(np.mean(f1s))
   
   f1_score = np.mean(f1_scores)
   
   return total_loss / len(loader), accuracy, digit_cms, digit_preds, digit_labels, f1_score

def evaluate(model, loader, criterion, device):
   model.eval()
   total_loss = 0
   total_correct = 0
   total_samples = 0
   digit_preds = [[], [], []]
   digit_labels = [[], [], []]
   
   with torch.no_grad():
       for images, labels, _ in loader:
           images = images.to(device)
           labels = labels.to(device)
           
           outputs = model(images)
           loss = criterion(outputs.view(-1, 10), labels.view(-1))
           
           _, predicted = outputs.max(2)
           correct = (predicted == labels).all(dim=1).sum().item()
           total_correct += correct
           total_samples += labels.size(0)
           total_loss += loss.item()
           
           for i in range(3):
               digit_preds[i].extend(predicted[:, i].cpu().numpy())
               digit_labels[i].extend(labels[:, i].cpu().numpy())
   
   accuracy = total_correct / total_samples
   digit_cms = [confusion_matrix(digit_labels[i], digit_preds[i], labels=range(10)) 
               for i in range(3)]
   
   f1_scores = []
   for i in range(3):
       cm = digit_cms[i]
       precisions = np.diag(cm) / (cm.sum(axis=0) + 1e-15)
       recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-15)
       f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
       f1_scores.append(np.mean(f1s))
   
   f1_score = np.mean(f1_scores)
   
   return total_loss / len(loader), accuracy, digit_cms, f1_score

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_type: str, log_file=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    epoch_start = datetime.now()
    
    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(2)
        correct = (predicted == labels).all(dim=1).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        
        if (i + 1) % max(1, len(train_loader) // 5) == 0:
            avg_loss = total_loss / (i + 1)
            accuracy = total_correct / total_samples
            logging.info(f'{model_type} - Epoch {epoch} - Batch {i+1}/{len(train_loader)} - '
                        f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    epoch_duration = datetime.now() - epoch_start
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_samples
    
    logging.info(f"{model_type} - Epoch {epoch} completed in {epoch_duration}")
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nEpoch {epoch} ({epoch_duration}):\n")
            f.write(f"Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}\n")
    
    return epoch_loss, epoch_acc

def plot_model_training_history(train_losses, train_accs, val_losses, val_accs, 
                            digit_cms, model_type, vis_dir):
    logging.info(f"Creating training history visualisation for {model_type}")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Model Loss ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title(f'Model Accuracy ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()
    
    titles = ['First Digit', 'Second Digit', 'Third Digit']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        sns.heatmap(digit_cms[i], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{titles[i]} ({model_type})')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrices.png'))
    plt.close()


def plot_training_metrics(metrics_dict, save_path):
    plt.figure(figsize=(15, 10))
    
    titles = ['Loss', 'Accuracy', 'F1 Score']
    metrics = ['losses', 'accuracies', 'val_f1s']
    
    for i, (title, metric) in enumerate(zip(titles, metrics)):
        plt.subplot(1, 3, i+1)
        for model_type, data in metrics_dict.items():
            plt.plot(data[metric], label=model_type, marker='o')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_final_comparison(final_metrics, save_path):
    metrics = ['Accuracy', 'F1 Score']
    model_types = list(final_metrics.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model_type in enumerate(model_types):
        values = [final_metrics[model_type]['accuracy'], 
                 final_metrics[model_type]['f1_score']]
        rects = ax.bar(x + i*width, values, width, label=model_type)
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Score')
    ax.set_title('Final Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(model_type: str, model, train_loader, val_loader, test_loader, device, 
               paths: dict, batch_size: int, epochs: int, learning_rate: float):
    model_path = paths[model_type]
    log_file = os.path.join(paths['base'], 'run_log.txt')
    
    if model_type == 'real_only':
        with open(log_file, 'a') as f:
            f.write(f"\n=== {model_type.upper()} Model Training ===\n")
            f.write(f"Number of training samples: {len(train_loader.dataset)}\n\n")
    else:
        with open(log_file, 'a') as f:
            f.write(f"\n=== {model_type.upper()} Model Training ===\n")
            real_samples = sum(1 for d in train_loader.dataset.datasets[0])
            synth_samples = sum(1 for d in train_loader.dataset.datasets[1])
            f.write(f"Number of real training samples: {real_samples}\n")
            f.write(f"Number of synthetic training samples: {synth_samples}\n\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    metrics = {
        'losses': [],
        'accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_f1s': []
    }
    
    best_val_acc = 0.0
    training_start = datetime.now()
    
    logging.info(f"Starting {model_type} model training")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_type, log_file)
        
        logging.info(f"{model_type} - Running validation for epoch {epoch}")
        val_loss, val_acc, val_cms, val_f1 = evaluate(model, val_loader, criterion, device)
        logging.info(f"{model_type} - Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        metrics['losses'].append(train_loss)
        metrics['accuracies'].append(train_acc)
        metrics['val_losses'].append(val_loss)
        metrics['val_accuracies'].append(val_acc)
        metrics['val_f1s'].append(val_f1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(model_path['base'], 'best_model.pth'))
        
        with open(log_file, 'a') as f:
            f.write(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}\n")
    
    logging.info(f"Running final test evaluation for {model_type} model")
    test_loss, test_acc, test_cms, test_f1 = evaluate(model, test_loader, criterion, device)
    
    plot_model_training_history(
        metrics['losses'], metrics['accuracies'],
        metrics['val_losses'], metrics['val_accuracies'],
        test_cms, model_type, model_path['vis']
    )
    
    training_duration = datetime.now() - training_start
    
    logging.info(f"{model_type} - Test Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    logging.info(f"Total training time: {training_duration}")
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== {model_type} Model Final Results ===\n")
        f.write(f"Training Time: {training_duration}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    
    return metrics, {'accuracy': test_acc, 'f1_score': test_f1}

def main():
    paths = setup_run_directory()
    device = torch_directml.device()
    
    batch_size = 256
    epochs = 25
    learning_rate = 0.001
    
    train_path = os.path.join('data', 'train')
    val_path = os.path.join('data', 'val')
    test_path = os.path.join('data', 'test')
    synthetic_path = os.path.join('scripts', 'Task5', 'GAN', 'results', 
                                 '20241227_005212', 'visualisations', 
                                 'samples', 'final_samples')
    
    train_dataset = TripleMNISTDataset(train_path, train=True)
    synthetic_dataset = TripleMNISTDataset(synthetic_path, train=True, is_synthetic=True)
    val_dataset = TripleMNISTDataset(val_path)
    test_dataset = TripleMNISTDataset(test_path)
    
    synthetic_subset_size = min(len(synthetic_dataset), len(train_dataset) // 4)
    real_subset_size = len(train_dataset) - synthetic_subset_size
    
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    real_subset = torch.utils.data.Subset(train_dataset, indices[:real_subset_size])
    
    synthetic_indices = list(range(len(synthetic_dataset)))
    np.random.shuffle(synthetic_indices)
    synthetic_subset = torch.utils.data.Subset(synthetic_dataset, synthetic_indices[:synthetic_subset_size])
    
    combined_dataset = ConcatDataset([real_subset, synthetic_subset])
    
    num_workers = 4
    pin_memory = True
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    initial_model = CNN().to(device)
    create_run_log_header(os.path.join(paths['base'], 'run_log.txt'),
                         initial_model, batch_size, epochs, learning_rate)
    
    all_metrics = {}
    final_metrics = {}
    
    real_loader = DataLoader(train_dataset, batch_size=batch_size, 
                           shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    model = CNN().to(device)
    metrics, final = train_model('real_only', model, real_loader,
                               val_loader, test_loader, device, paths,
                               batch_size, epochs, learning_rate)
    all_metrics['real_only'] = metrics
    final_metrics['real_only'] = final
    
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    model = CNN().to(device)
    metrics, final = train_model('combined', model, combined_loader,
                               val_loader, test_loader, device, paths,
                               batch_size, epochs, learning_rate)
    all_metrics['combined'] = metrics
    final_metrics['combined'] = final
    
    plot_training_metrics(all_metrics, os.path.join(paths['root_vis'], 'training_comparison.png'))
    plot_final_comparison(final_metrics, os.path.join(paths['root_vis'], 'final_comparison.png'))
    
    logging.info("All training completed successfully!")

if __name__ == "__main__":
    main()