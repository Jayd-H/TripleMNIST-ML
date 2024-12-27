# Jayden Holdsworth 2024
# Task 5: GAN
# Generates full 84x84 images of triple digit combinations from the MNIST dataset
# Discriminator processes each digit separately and combines the features for final classification

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from PIL import Image
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import torch_directml # type: ignore
import random
import shutil
import glob

#* editor was bugging without the type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def setup_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join('scripts', 'Task5', 'GAN', 'results', timestamp)
    vis_dir = os.path.join(base_dir, 'visualisations')
    sample_dir = os.path.join(vis_dir, 'samples')
    grid_dir = os.path.join(vis_dir, 'grids')
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    
    return base_dir, vis_dir, sample_dir, grid_dir

def create_run_log_header(filepath: str, generator, discriminator, batch_size: int, epochs: int, g_lr: float, d_lr: float):
    with open(filepath, 'w') as f:
        f.write("=== GAN Training Run Log ===\n\n")
        f.write("This log file contains the following information:\n")
        f.write("1. Configuration parameters\n")
        f.write("2. Per-epoch training metrics\n")
        f.write("3. Sample generation metrics\n\n")
        f.write("Metrics explanation:\n")
        f.write("- D Loss: Discriminator loss (real + fake + auxiliary)\n")
        f.write("- G Loss: Generator loss (adversarial + auxiliary)\n\n")
        f.write("=== Configuration ===\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {epochs}\n")
        f.write(f"Generator Learning Rate: {g_lr}\n")
        f.write(f"Discriminator Learning Rate: {d_lr}\n")
        f.write("\nGenerator Architecture:\n")
        f.write(str(generator))
        f.write("\n\nDiscriminator Architecture:\n")
        f.write(str(discriminator))
        f.write("\n\n=== Training Log ===\n")

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(10, 16)
        
        self.noise_proj = nn.Linear(latent_dim, 256 * 6 * 6)
        self.label_proj = nn.Linear(16 * 3, 128 * 6 * 6)
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 7, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        batch_size = noise.size(0)
        #print(f"\nGenerator Forward Pass:")
        #print(f"Input noise shape: {noise.shape}")
        #print(f"Input labels shape: {labels.shape}")
        
        # Process labels
        emb1 = self.label_emb(labels[:, 0])
        emb2 = self.label_emb(labels[:, 1])
        emb3 = self.label_emb(labels[:, 2])
        emb_cat = torch.cat([emb1, emb2, emb3], dim=1)
        #print(f"Combined label embeddings shape: {emb_cat.shape}")
        
        noise_feat = self.noise_proj(noise.view(batch_size, -1))
        label_feat = self.label_proj(emb_cat)
        #print(f"Projected noise shape: {noise_feat.shape}")
        #print(f"Projected label shape: {label_feat.shape}")
        
        noise_feat = noise_feat.view(batch_size, 256, 6, 6)
        label_feat = label_feat.view(batch_size, 128, 6, 6)
        combined = torch.cat([noise_feat, label_feat], dim=1)
        #print(f"Combined features shape: {combined.shape}")
        
        output = self.conv_blocks(combined)
        #print(f"Final output shape: {output.shape}")
        return output

class Discriminator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(10, 16)
        self.label_proj = nn.Sequential(
            nn.Linear(16 * 3, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        self.features_per_section = 21 * 7 * 64
        
        self.digit_classifier = nn.Sequential(
            nn.Linear(self.features_per_section, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
        
        self.adversarial = nn.Sequential(
            nn.Linear(self.features_per_section * 3 + 64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        batch_size = img.size(0)
        
        emb1 = self.label_emb(labels[:, 0])
        emb2 = self.label_emb(labels[:, 1])
        emb3 = self.label_emb(labels[:, 2])
        label_feat = self.label_proj(torch.cat([emb1, emb2, emb3], dim=1))
        
        splits = torch.chunk(img, 3, dim=3)
        
        features = []
        digit_preds = []
        
        for i, split in enumerate(splits):
            feat = self.conv_blocks(split)
            feat_flat = feat.view(batch_size, -1)
            features.append(feat_flat)
            digit_preds.append(self.digit_classifier(feat_flat))
        
        combined_features = torch.cat(features + [label_feat], dim=1)
        validity = self.adversarial(combined_features)
        
        return validity.squeeze(), torch.stack(digit_preds, dim=1)

class TripleMNISTDataset(Dataset):
    def __init__(self, data_path: str):
        self.images = []
        self.labels = []
        
        logging.info(f"Loading dataset from {data_path}")
        
        data_folders = ['train', 'val', 'test']
        
        for folder in data_folders:
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                folder_combinations = sorted([d for d in os.listdir(folder_path) 
                                           if os.path.isdir(os.path.join(folder_path, d)) 
                                           and len(d) == 3 and d.isdigit()])
                
                for digit_combo in folder_combinations:
                    combo_dir = os.path.join(folder_path, digit_combo)
                    image_files = sorted([f for f in os.listdir(combo_dir) if f.endswith('.png')])
                    
                    label = torch.tensor([int(d) for d in digit_combo], dtype=torch.long)
                    
                    for img_file in image_files:
                        img_path = os.path.join(combo_dir, img_file)
                        self.images.append(img_path)
                        self.labels.append(label)
        
        logging.info(f"Successfully loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)
        return img * 2 - 1, self.labels[idx]

def train_one_epoch(generator, discriminator, g_optimizer, d_optimizer, dataloader, device, latent_dim, epoch, num_epochs, log_file=None):
    generator.train()
    discriminator.train()
    
    aux_weight = 1
    
    total_d_loss = 0
    total_g_loss = 0
    batches = len(dataloader)
    epoch_start = datetime.now()
    
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)
        
        d_optimizer.zero_grad()
        
        label_real = torch.ones(batch_size, device=device) * 0.9
        label_fake = torch.zeros(batch_size, device=device) + 0.1
        
        validity_real, digits_real = discriminator(real_images, labels)
        d_real_loss = F.binary_cross_entropy(validity_real, label_real)
        d_real_aux_loss = sum(F.cross_entropy(digits_real[:, i], labels[:, i]) for i in range(3)) / 3.0
        
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise, labels)
        validity_fake, digits_fake = discriminator(fake_images.detach(), labels)
        d_fake_loss = F.binary_cross_entropy(validity_fake, label_fake)
        
        d_loss = d_real_loss + d_fake_loss + aux_weight * d_real_aux_loss
        d_loss.backward()
        d_optimizer.step()
        
        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise, labels)
        validity, digits = discriminator(fake_images, labels)
            
        g_adv_loss = F.binary_cross_entropy(validity, label_real)
        g_aux_loss = sum(F.cross_entropy(digits[:, i], labels[:, i]) for i in range(3)) / 3.0
            
        g_loss = g_adv_loss + aux_weight * g_aux_loss
        g_loss.backward()
        g_optimizer.step()
        
        
            
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        
        if (i + 1) % max(1, batches // 10) == 0:
            avg_d_loss = total_d_loss / (i + 1)
            avg_g_loss = total_g_loss / (i + 1)
            logging.info(
                f'Epoch [{epoch}/{num_epochs}] Batch [{i+1}/{batches}] - '
                f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}'
            )
    
    epoch_duration = datetime.now() - epoch_start
    epoch_d_loss = total_d_loss / batches
    epoch_g_loss = total_g_loss / batches
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nEpoch {epoch} (Duration: {epoch_duration}):\n")
            f.write(f"D Loss: {epoch_d_loss:.4f}\n")
            f.write(f"G Loss: {epoch_g_loss:.4f}\n")
            f.write(f"G/D Ratio: {epoch_g_loss/epoch_d_loss:.4f}\n")
    
    return epoch_d_loss, epoch_g_loss

def generate_random_combinations(num_combinations):
    all_possible = [f"{i:03d}" for i in range(1000)]
    return random.sample(all_possible, num_combinations)

def plot_training_history(g_losses_epoch, d_losses_epoch, output_dir):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    epochs = range(1, len(g_losses_epoch) + 1)
    plt.plot(epochs, g_losses_epoch, label='Generator Loss', marker='o')
    plt.plot(epochs, d_losses_epoch, label='Discriminator Loss', marker='o')
    plt.title('GAN Training History - Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    loss_ratio = [g/d for g, d in zip(g_losses_epoch, d_losses_epoch)]
    plt.plot(epochs, loss_ratio, label='G/D Loss Ratio', color='green', marker='o')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimal Ratio')
    plt.title('Generator/Discriminator Loss Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_sample_generated_images(data_dir, output_dir):
    plt.figure(figsize=(12, 16))
    
    digit_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    selected_dirs = random.sample(digit_dirs, 4)
    
    for row, digit_combo in enumerate(selected_dirs):
        combo_dir = os.path.join(data_dir, digit_combo)
        image_files = sorted(glob.glob(os.path.join(combo_dir, "*.png")))
        selected_files = random.sample(image_files, 3)
        
        plt.figtext(0.5, 0.92 - (row * 0.22), f"Sample {digit_combo}", 
                   ha='center', va='bottom', fontsize=14)
        
        for col, img_path in enumerate(selected_files):
            plt.subplot(4, 3, row * 3 + col + 1)
            img = Image.open(img_path).convert('L')
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.2)
    plt.savefig(os.path.join(output_dir, 'sample_generated.png'), 
                bbox_inches='tight', pad_inches=0.5)
    plt.close()

def save_generated_images(generator, device, output_dir, latent_dim, num_combinations=50, images_per_combination=100):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    generator.eval()
    logging.info(f"Generating {num_combinations} digit combinations, {images_per_combination} images each")
    
    new_combinations = generate_random_combinations(num_combinations)
    
    with torch.no_grad():
        for idx, combo in enumerate(new_combinations):
            combo_dir = os.path.join(output_dir, combo)
            os.makedirs(combo_dir, exist_ok=True)
            
            labels = torch.tensor([[int(d) for d in combo] for _ in range(images_per_combination)],
                                dtype=torch.long, device=device)
            
            noise = torch.randn(images_per_combination, latent_dim, device=device)
            fakes = generator(noise, labels)
            fakes = (fakes + 1) / 2
            
            for i, fake in enumerate(fakes):
                img = Image.fromarray((fake.squeeze().cpu().numpy() * 255).astype(np.uint8))
                img.save(os.path.join(combo_dir, f"{i}_{combo}.png"))
            
            if (idx + 1) % 10 == 0:
                logging.info(f"Generated images for {idx + 1}/{num_combinations} combinations")
                
def generate_fixed_grid_samples(generator, device, latent_dim, grid_dir, epoch):
    generator.eval()
    fixed_combinations = [
        "000", "111", "222", "333", "444", "555", "666", "777", "888", "999",  # Same digits
        "012", "123", "234", "345", "456", "567", "678", "789", "890", "901",  # Sequential
        "147", "258", "369", "741", "852", "963", "159", "357", "048", "246",  # Patterns
        "100", "200", "300", "400", "500", "600", "700", "800", "900", "110",  # First digit varies
        "010", "020", "030", "040", "050", "060", "070", "080", "090", "011",  # Second digit varies
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "101",  # Third digit varies
        "135", "246", "357", "468", "579", "680", "791", "802", "913", "024",  # Mixed patterns
        "111", "222", "333", "444", "555", "666", "777", "888", "999", "000",  # Repeated (for stability check)
        "123", "234", "345", "456", "567", "678", "789", "890", "901", "012",  # Sequential (different start)
        "159", "267", "348", "456", "567", "678", "789", "890", "901", "012"   # More variety
    ]
    
    with torch.no_grad():
        fixed_noise = torch.randn(100, latent_dim, device=device)
        labels = torch.tensor([[int(d) for d in combo] for combo in fixed_combinations],
                            dtype=torch.long, device=device)
        
        fake_images = generator(fixed_noise, labels)
        fake_images = (fake_images + 1) / 2
        
        plt.figure(figsize=(20, 20))
        for idx, (img, combo) in enumerate(zip(fake_images, fixed_combinations)):
            plt.subplot(10, 10, idx + 1)
            plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            plt.title(combo, fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(grid_dir, f'grid_epoch_{epoch:03d}.png'), dpi=100)
        plt.close()

def create_training_gif(vis_dir, grid_dir):
    logging.info("Creating training progression GIF...")
    grid_images = sorted(glob.glob(os.path.join(grid_dir, 'grid_epoch_*.png')))
    
    images = []
    for image_path in grid_images:
        images.append(Image.open(image_path))
    
    gif_path = os.path.join(vis_dir, 'training_progression.gif')
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    logging.info(f"GIF saved to {gif_path}")

def main():
    base_dir, vis_dir, sample_dir, grid_dir = setup_run_directory()
    log_file = os.path.join(base_dir, 'run_log.txt')
    
    device = torch_directml.device()
    
    latent_dim = 128
    batch_size = 256
    num_epochs = 200
    g_lr = 0.00005
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    create_run_log_header(log_file, generator, discriminator, batch_size, num_epochs, g_lr, d_lr)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))
    
    dataset = TripleMNISTDataset('data')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=8, pin_memory=True)
    
    g_losses_epoch = []
    d_losses_epoch = []
    
    training_start_time = datetime.now()
    logging.info("Starting training")
    
    logging.info("Generating initial state visualization...")
    generate_fixed_grid_samples(generator, device, latent_dim, grid_dir, 0)
    
    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        d_loss, g_loss = train_one_epoch(
            generator, discriminator, g_optimizer, d_optimizer,
            dataloader, device, latent_dim, epoch + 1, num_epochs,
            log_file=log_file
        )
        
        g_losses_epoch.append(g_loss)
        d_losses_epoch.append(d_loss)
        
        logging.info(f"Generating epoch {epoch + 1} visualization grid...")
        generate_fixed_grid_samples(generator, device, latent_dim, grid_dir, epoch + 1)
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Generating epoch {epoch + 1} sample images...")
            sample_epoch_dir = os.path.join(sample_dir, f'epoch_{epoch+1}')
            save_generated_images(generator, device, sample_epoch_dir, latent_dim,
                               num_combinations=10, images_per_combination=5)
        
        epoch_duration = datetime.now() - epoch_start
        logging.info(f"Epoch {epoch + 1} completed in {epoch_duration}")
        logging.info(f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, G/D Ratio: {g_loss/d_loss:.4f}")
    
    training_duration = datetime.now() - training_start_time
    
    logging.info("Training completed. Generating final outputs...")
    
    with open(log_file, 'a') as f:
        f.write("\n=== Final Results ===\n")
        f.write(f"Total Training Time: {training_duration}\n")
        f.write(f"Final G Loss: {g_losses_epoch[-1]:.4f}\n")
        f.write(f"Final D Loss: {d_losses_epoch[-1]:.4f}\n")
        f.write(f"Final G/D Ratio: {g_losses_epoch[-1]/d_losses_epoch[-1]:.4f}\n")
    
    logging.info("Generating final synthetic dataset...")
    final_samples_dir = os.path.join(sample_dir, 'final_samples')
    save_generated_images(generator, device, final_samples_dir, latent_dim)
    
    logging.info("Creating training visualizations...")
    plot_training_history(g_losses_epoch, d_losses_epoch, vis_dir)
    plot_sample_generated_images(final_samples_dir, vis_dir)
    
    logging.info("Creating training progression GIF...")
    create_training_gif(vis_dir, grid_dir)
    
    logging.info(f"Training completed in {training_duration}")
    logging.info(f"All results saved to {base_dir}")
    
    torch.save({
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'g_losses': g_losses_epoch,
        'd_losses': d_losses_epoch,
        'training_duration': training_duration,
        'final_epoch': num_epochs
    }, os.path.join(base_dir, 'final_model_state.pth'))
    
    logging.info("Run completed successfully!")

if __name__ == "__main__":
    main()