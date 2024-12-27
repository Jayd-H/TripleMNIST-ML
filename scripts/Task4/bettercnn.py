# Jayden Holdsworth 2024
# Task 4: CNN
# Algorithm to train a CNN model on the provided triple MNIST dataset
# Treats each image as three digits with 10 different outputs
# Similar to Task3 CNN but with data augmentation and dropout

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
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
    base_dir = os.path.join('scripts', 'Task4', 'results', timestamp)
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
    def __init__(self, data_path: str, train=False):
        self.images = []
        self.labels = []
        self.image_names = []
        self.train = train
        
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        
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
                self.labels.append([int(d) for d in digit_combo])
                self.image_names.append(img_file)
        
        logging.info(f"Successfully loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        
        if self.train:
            img = self.transform(img)
        
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        
        width = img.shape[1]
        split_width = width // 3
        
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

def plot_feature_maps(model, sample_image, feature_map_dir, epoch):
    if epoch == 1 or epoch % 5 == 0:
        logging.info(f"Creating feature map visualisation for epoch: {epoch}")
        model.eval()
        with torch.no_grad():
            # Add batch dimension if it's not present
            if len(sample_image.size()) == 3:
                sample_image = sample_image.unsqueeze(0)
                
            width = sample_image.size(3)
            split_width = width // 3
            
            digit1 = sample_image[:, :, :, :split_width]
            features = model.features(digit1)
        
        feature_map = features[0, 0].cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature Map (First Digit) at Epoch {epoch}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(feature_map_dir, f'feature_map_epoch_{epoch}.png'))
        plt.close()

def plot_error_distribution(digit_labels, digit_preds, vis_dir):
    logging.info("Creating error distribution visualisation")
    
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        errors = [abs(true - pred) for true, pred in zip(digit_labels[i], digit_preds[i])]
        plt.hist(errors, bins=10, edgecolor='black')
        plt.title(f'Distribution of Prediction Errors - Digit {i+1}')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
    plt.close()

def plot_training_history(train_losses, train_accs, val_losses, val_accs, digit_cms, vis_dir):
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
    
    titles = ['First Digit', 'Second Digit', 'Third Digit']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        sns.heatmap(digit_cms[i], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrices.png'))
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
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(2)
        correct = (predicted == labels).all(dim=1).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        
        total_loss += loss.item()
        
        if (i + 1) % max(1, batches // 10) == 0:
            accuracy = total_correct / total_samples
            avg_loss = total_loss / (i + 1)
            logging.info(f'Epoch {epoch}: Batch {i+1}/{batches} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    if sample_image is not None and feature_map_dir is not None:
        plot_layer_activations(model, sample_image, feature_map_dir, epoch)
        generate_attention_maps(model, loader, feature_map_dir, epoch)
    
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
    digit_cms = [confusion_matrix(digit_labels[i], digit_preds[i], labels=range(10)) for i in range(3)]
    
    f1_scores = []
    for i in range(3):
        cm = digit_cms[i]
        precisions = np.diag(cm) / (cm.sum(axis=0) + 1e-15)
        recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-15)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
        f1_scores.append(np.mean(f1s))
    
    f1_score = np.mean(f1_scores)
    
    return total_loss / len(loader), accuracy, digit_cms, digit_preds, digit_labels, f1_score

def plot_layer_activations(model, sample_image, feature_map_dir, epoch):
    """
    Visualises activations from all three convolutional layers for a sample image.
    Shows how the image features are processed through the network.
    """
    if epoch == 1 or epoch % 5 == 0:
        logging.info(f"Creating layer activation visualisations for epoch: {epoch}")
        model.eval()
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output)
        
        hooks = []
        for layer in [0, 5, 10]:
            hooks.append(model.features[layer].register_forward_hook(hook_fn))
        
        with torch.no_grad():
            if len(sample_image.size()) == 3:
                sample_image = sample_image.unsqueeze(0)
            
            width = sample_image.size(3)
            split_width = width // 3
            digit1 = sample_image[:, :, :, :split_width]
            _ = model.features(digit1)
        
        for hook in hooks:
            hook.remove()
        
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 4, width_ratios=[1, 0.4, 0.4, 0.4], 
                            hspace=0.2, wspace=0.2)
        
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(digit1[0, 0].cpu().numpy(), cmap='gray')
        ax_orig.set_title('Input Image')
        ax_orig.axis('off')
        
        explanations = [
            "Conv Layer 1: Detects basic features like\nedges and simple patterns",
            "Conv Layer 2: Combines basic features into\nmore complex patterns",
            "Conv Layer 3: Creates high-level abstract\nrepresentations of digit parts"
        ]
        
        for i, explanation in enumerate(explanations):
            ax_text = fig.add_subplot(gs[i+1, 0])
            ax_text.text(0.1, 0.5, explanation, 
                        wrap=True, horizontalalignment='left',
                        verticalalignment='center')
            ax_text.axis('off')
        
        for layer_idx, activation in enumerate(activations):
            for feat_idx in range(3):
                ax = fig.add_subplot(gs[layer_idx + 1, feat_idx + 1])
                feature_map = activation[0, feat_idx].cpu().numpy()
                im = ax.imshow(feature_map, cmap='viridis')
                ax.axis('off')
                
                if feat_idx == 2:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Feature Intensity', rotation=270, labelpad=15)
        
        plt.suptitle(f'Feature Extraction Through Network at Epoch {epoch}', size=16, y=0.95)
        plt.savefig(os.path.join(feature_map_dir, f'layer_activations_epoch_{epoch}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
def visualise_augmentations(dataset, vis_dir):
    """
    Creates a grid showing the same image with different augmentations applied.
    Demonstrates the effect of RandomAffine and RandomPerspective transforms.
    """
    logging.info("Creating data augmentation visualisation")
    
    sample_image, _, image_name = dataset[0]
    
    plt.figure(figsize=(15, 10))
    rows, cols = 3, 4
    
    plt.subplot(rows, cols, 1)
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for i in range(1, rows * cols):
        augmented = dataset.transform(Image.fromarray(
            (sample_image.squeeze().numpy() * 255).astype(np.uint8)
        ))
        augmented = np.array(augmented).astype(np.float32) / 255.0
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(augmented, cmap='gray')
        plt.title(f'Augmentation {i}')
        plt.axis('off')
    
    plt.suptitle('Sample Data Augmentations', size=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'augmentation_examples.png'))
    plt.close()
    
def analyse_model_confidence(model, loader, device, vis_dir):
    """
    Analyses and visualises the model's confidence in its predictions,
    comparing confidence levels for correct vs incorrect predictions.
    """
    logging.info("Analysing model confidence patterns")
    model.eval()
    
    confidences_correct = [[], [], []]
    confidences_incorrect = [[], [], []]
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=2)
            
            max_probs, predictions = torch.max(probabilities, dim=2)
            
            for digit_pos in range(3):
                correct_mask = predictions[:, digit_pos] == labels[:, digit_pos]
                
                confidences_correct[digit_pos].extend(
                    max_probs[:, digit_pos][correct_mask].cpu().numpy()
                )
                confidences_incorrect[digit_pos].extend(
                    max_probs[:, digit_pos][~correct_mask].cpu().numpy()
                )
    
    plt.figure(figsize=(12, 6))
    positions = np.arange(1, 7, 2)
    digits = ['First Digit', 'Second Digit', 'Third Digit']
    
    violin_parts = plt.violinplot(
        confidences_correct, 
        positions=[p - 0.3 for p in positions], 
        showmeans=True
    )
    for pc in violin_parts['bodies']:
        pc.set_facecolor('green')
        pc.set_alpha(0.5)
    
    violin_parts = plt.violinplot(
        confidences_incorrect, 
        positions=[p + 0.3 for p in positions],
        showmeans=True
    )
    for pc in violin_parts['bodies']:
        pc.set_facecolor('red')
        pc.set_alpha(0.5)
    
    plt.xticks(positions, digits)
    plt.ylabel('Confidence Score')
    plt.title('Model Confidence Distribution for Correct vs Incorrect Predictions')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Correct Predictions'),
        Patch(facecolor='red', alpha=0.5, label='Incorrect Predictions')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confidence_analysis.png'))
    plt.close()
    
    for i, digit in enumerate(digits):
        correct_mean = np.mean(confidences_correct[i])
        incorrect_mean = np.mean(confidences_incorrect[i])
        logging.info(f"{digit} digit confidence stats:")
        logging.info(f"  Mean confidence when correct: {correct_mean:.4f}")
        logging.info(f"  Mean confidence when incorrect: {incorrect_mean:.4f}")
        logging.info(f"  Confidence gap: {correct_mean - incorrect_mean:.4f}")
        
def generate_attention_maps(model, loader, feature_map_dir, epoch):
    """
    Generates attention heat maps showing which parts of each digit
    the model focuses on most strongly. Creates a 2x2 grid with reduced spacing.
    """
    if epoch == 1 or epoch % 5 == 0:
        logging.info(f"Generating attention heat maps for epoch: {epoch}")
        model.eval()
        
        def get_activation(name):
            activation = {}
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook, activation
        
        hook, activation = get_activation('features')
        handle = model.features[-3].register_forward_hook(hook)
        
        with torch.no_grad():
            samples = []
            labels = []
            for images, label, _ in loader:
                samples.append(images[0])
                labels.append(''.join(map(str, label[0].tolist())))
                if len(samples) == 2:
                    break
            
            fig = plt.figure(figsize=(8, 5))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.1, hspace=0.3)
            
            for sample_idx, (sample_image, label) in enumerate(zip(samples, labels)):
                sample_image = sample_image.to(next(model.parameters()).device)
                if len(sample_image.size()) == 3:
                    sample_image = sample_image.unsqueeze(0)
                
                width = sample_image.size(3)
                split_width = width // 3
                
                ax = fig.add_subplot(gs[0, sample_idx])
                plt.imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
                plt.title(f'Original Image (Label: {label})')
                plt.axis('off')
                
                ax = fig.add_subplot(gs[1, sample_idx])
                
                attention_maps = []
                for i in range(3):
                    digit = sample_image[:, :, :, i*split_width:(i+1)*split_width]
                    _ = model.features(digit)
                    feature_maps = activation['features']
                    attention_map = torch.mean(feature_maps, dim=1).squeeze().cpu().numpy()
                    
                    from scipy.ndimage import zoom
                    zoom_factor = digit.shape[2] / attention_map.shape[0]
                    attention_maps.append(zoom(attention_map, zoom_factor))
                
                plt.imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
                combined_attention = np.concatenate(attention_maps, axis=1)
                plt.imshow(combined_attention, cmap='hot', alpha=0.5)
                plt.title(f'Attention Heat Map')
                plt.axis('off')
            
            plt.suptitle(f'Attention Analysis at Epoch {epoch}', size=16, y=1.0)
            plt.savefig(os.path.join(feature_map_dir, f'attention_maps_epoch_{epoch}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
        
        handle.remove()

def main():
    base_dir, vis_dir, feature_map_dir = setup_run_directory()
    log_file = os.path.join(base_dir, 'run_log.txt')
    
    device = torch_directml.device()
    logging.info("Initialising training with DirectML (AMD GPU acceleration)")
    
    batch_size = 256
    epochs = 250
    learning_rate = 0.001
    
    train_dataset = TripleMNISTDataset('data/train', train=True)
    val_dataset = TripleMNISTDataset('data/val')
    test_dataset = TripleMNISTDataset('data/test')
    
    visualise_augmentations(train_dataset, vis_dir)
    
    sample_image, _, _ = train_dataset[0]
    sample_image = sample_image.to(device)
    
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
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
        val_loss, val_acc, val_cms, _, _, val_f1 = evaluate(model, val_loader, criterion, device)
        
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
        logging.info(f"Validation F1 Score: {val_f1:.4f}")
    
    training_duration = datetime.now() - training_start_time
    
    logging.info("Starting final test evaluation")
    test_loss, test_acc, test_cms, test_preds, test_labels, test_f1 = evaluate(model, test_loader, criterion, device)
    
    analyse_model_confidence(model, test_loader, device, vis_dir)
    
    logging.info("Creating visualisations")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, test_cms, vis_dir)
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