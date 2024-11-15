import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dirs = img_dirs  # List of directories
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Look up the image name
        img_name = self.data.iloc[idx]['image_id'] + '.jpg'
        
        # Search for the image in the directories
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break
        else:
            raise FileNotFoundError(f"Image {img_name} not found in specified directories.")
        
        # Get the label
        label = self.data.iloc[idx]['dx']  # Diagnosis column
        label_map = {label: idx for idx, label in enumerate(self.data['dx'].unique())}
        label = label_map[label]

        if self.transform:
            image = self.transform(image)

        return image, label

metadata_path = "../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_path)

# Check the number of unique images in metadata
print(f"Total images in metadata: {len(metadata)}")

# Split metadata into train and test sets
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

# Save split metadata for easier loading
train_metadata.to_csv("train_metadata.csv", index=False)
test_metadata.to_csv("test_metadata.csv", index=False)

# Directories containing images
image_dirs = [
    "../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1",
    "../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 256x256
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Datasets
train_dataset = HAM10000Dataset(csv_file="train_metadata.csv", img_dirs=image_dirs, transform=transform)
test_dataset = HAM10000Dataset(csv_file="test_metadata.csv", img_dirs=image_dirs, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Print dataset sizes
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# Visualize one batch of images
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

# Display first 4 images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    axes[i].imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Denormalize
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")
plt.show()

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)

        proj_value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 8  # Downsample by 8 (adjusted for 64x64 output)
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            SelfAttention(128),  # Self-Attention after first upscale
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SelfAttention(32),  # Self-Attention in the middle layers
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        out = self.upsample(out)
        img = self.final_layer(out)
        return img

# Instantiate the generator
latent_dim = 100  # Size of latent vector
img_channels = 3  # RGB images
img_size = 64  # Output image size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim, img_channels, img_size).to(device)

# Test the generator
z = torch.randn(4, latent_dim).to(device)  # Random latent vector (batch size = 4)
generated_images = generator(z)

print(f"Generated image shape: {generated_images.shape}")  # Should be [4, 3, 64, 64]

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)

        proj_value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if downsample else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu(out)
        out = self.pool(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            ResidualBlock(img_channels, 64, downsample=True),            # 64x64 -> 32x32
            SelfAttention(64),
            ResidualBlock(64, 128, downsample=True),           # 32x32 -> 16x16
            SelfAttention(128),
            ResidualBlock(128, 256, downsample=True),          # 16x16 -> 8x8
            ResidualBlock(256, 512, downsample=True)           # 8x8 -> 4x4
        )

        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Final score output
            nn.Sigmoid()  # Outputs probability of being real or fake
        )

    def forward(self, img):
        out = self.model(img)
        out = self.final_layer(out)
        return out

# Instantiate the discriminator
img_channels = 3  # RGB images
img_size = 64  # Input image size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator(img_channels, img_size).to(device)

# Test the discriminator
batch_size = 4
test_images = torch.randn(batch_size, img_channels, img_size, img_size).to(device)  # Fake images batch
output = discriminator(test_images)

print(f"Discriminator output shape: {output.shape}")  # Should be [4, 1]

import warnings
warnings.filterwarnings("ignore")

from torch import optim
from tqdm import tqdm

def train_gan(generator, discriminator, train_loader, latent_dim, device, epochs=1000, lr=0.0002, beta1=0.5, beta2=0.999):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for real_images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            
            with torch.cuda.amp.autocast():  # Mixed precision training
                generated_images = generator(z)
                g_loss = criterion(discriminator(generated_images), valid)

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
            epoch_loss_G += g_loss.item()

            # Train Discriminator
            optimizer_D.zero_grad()
            with torch.cuda.amp.autocast():
                real_loss = criterion(discriminator(real_images), valid)
                fake_loss = criterion(discriminator(generated_images.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
            epoch_loss_D += d_loss.item()

            # Clear cache to reduce memory fragmentation
            torch.cuda.empty_cache()

        print(f"Epoch [{epoch+1}/{epochs}] | Generator Loss: {epoch_loss_G:.4f} | Discriminator Loss: {epoch_loss_D:.4f}")

    print("Training completed.")


# Call the train_gan function with the train_loader, generator, and discriminator
train_gan(generator, discriminator, train_loader, latent_dim, device)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import warnings
warnings.filterwarnings("ignore")

# Define EfficientNetV2 model for HAM10000 with 7 classes
class EfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes=7):  # 7 classes for HAM10000
        super(EfficientNetV2Classifier, self).__init__()
        self.efficientnet_v2 = models.efficientnet_v2_s(pretrained=True)
        
        in_features = self.efficientnet_v2.classifier[1].in_features
        self.efficientnet_v2.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet_v2(x)

# Initialize the model
model_EfficientNetV2 = EfficientNetV2Classifier(num_classes=7)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_EfficientNetV2.parameters(), lr=0.001)
epochs = 20

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            print("done")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = total_correct / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Validation after each epoch
        validate_model(model, test_loader)

# Validation loop
def validate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

model_EfficientNetV2.to(device)

# Train the model
train_model(model_EfficientNetV2, train_loader, test_loader, criterion, optimizer, epochs=epochs)

# Define ShuffleNetV2 model for HAM10000 with 7 classes
class ShuffleNetV2Classifier(nn.Module):
    def __init__(self, num_classes=7):  # 7 classes for HAM10000
        super(ShuffleNetV2Classifier, self).__init__()
        self.shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=True)
        
        # Modify the last fully connected layer to match the number of classes
        in_features = self.shufflenet_v2.fc.in_features
        self.shufflenet_v2.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.shufflenet_v2(x)

# Initialize the model
model_ShuffleNetV2 = ShuffleNetV2Classifier(num_classes=7)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ShuffleNetV2.parameters(), lr=0.001)
epochs = 20

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = total_correct / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Validation after each epoch
        validate_model(model, test_loader)

# Validation loop
def validate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

model_ShuffleNetV2.to(device)

# Train the model
train_model(model_ShuffleNetV2, train_loader, test_loader, criterion, optimizer, epochs=epochs)

def create_support_set(generator, model_EfficientNetV2, model_ShuffleNetV2, labels, noise_dim=128):
    noise = torch.randn(batch_size, noise_dim)  # Random noise for generator
    created_imgs = generator(noise, labels) 
    EfficientNetV2Classifier_labels = model_EfficientNetV2(created_imgs)
    ShuffleNetV2Classifier_labels = model_ShuffleNetV2(created_imgs)
    if EfficientNetV2Classifier_labels == labels and ShuffleNetV2Classifier_labels == labels:
        return created_imgs
    else:
        return None

import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, base_features=64):
        super(CNNEncoder, self).__init__()
        
        # Encoder block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Reduces 64x64 -> 32x32
        )
        
        # Encoder block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Reduces 32x32 -> 16x16
        )
        
        # Encoder block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Reduces 16x16 -> 8x8
        )
        
        # Encoder block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 8, base_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Reduces 8x8 -> 4x4
        )

    def forward(self, x):
        # Apply each encoder block to the input
        x = self.block1(x)  # 64x64 -> 32x32
        x = self.block2(x)  # 32x32 -> 16x16
        x = self.block3(x)  # 16x16 -> 8x8
        x = self.block4(x)  # 8x8 -> 4x4
        return x

import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(AttentionModule, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Linear transformations for multi-head attention
        self.query_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # Multi-head attention mechanism
        self.attn_heads = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1),
                nn.Softmax(dim=-1)  # Softmax across the spatial dimension
            ) for _ in range(num_heads)]
        )
        
        # Channel attention to recalibrate feature maps
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 16, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention to emphasize important regions in the spatial dimension
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Final 1x1 conv to combine outputs
        self.output_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
    
    def forward(self, features):
        # Compute query, key, and value maps for multi-head attention
        queries = self.query_conv(features)  # [B, C, H, W]
        keys = self.key_conv(features)       # [B, C, H, W]
        values = self.value_conv(features)   # [B, C, H, W]
        
        B, C, H, W = queries.size()
        queries = queries.view(B, self.num_heads, self.head_dim, H * W)  # [B, heads, head_dim, H*W]
        keys = keys.view(B, self.num_heads, self.head_dim, H * W)        # [B, heads, head_dim, H*W]
        values = values.view(B, self.num_heads, self.head_dim, H * W)    # [B, heads, head_dim, H*W]
        
        # Multi-head attention
        attention_outputs = []
        for i in range(self.num_heads):
            attn_weights = torch.bmm(queries[:, i], keys[:, i].transpose(1, 2))  # [B, head_dim, head_dim]
            attn_weights = self.attn_heads[i](attn_weights.view(B, self.head_dim, H, W))  # Apply learned attention map
            attn_output = torch.bmm(attn_weights.view(B, self.head_dim, H * W), values[:, i])  # [B, head_dim, H*W]
            attention_outputs.append(attn_output.view(B, self.head_dim, H, W))
        
        # Concatenate all attention head outputs
        multi_head_output = torch.cat(attention_outputs, dim=1)  # [B, C, H, W]
        
        # Channel Attention
        channel_attn_weights = self.channel_attention(multi_head_output)
        channel_attn_output = multi_head_output * channel_attn_weights  # Element-wise multiplication (recalibration)
        
        # Spatial Attention
        avg_pool = torch.mean(channel_attn_output, dim=1, keepdim=True)  # Average pooling across channels
        max_pool = torch.max(channel_attn_output, dim=1, keepdim=True)[0]  # Max pooling across channels
        spatial_attn_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        spatial_attn_output = channel_attn_output * spatial_attn_weights  # Element-wise multiplication (spatial recalibration)
        
        # Final 1x1 conv to produce the final attention output
        output = self.output_conv(spatial_attn_output)
        return output

import torch
import torch.nn as nn

class MTUNet2(nn.Module):
    def __init__(self, in_channels=3, base_features=64, num_classes=5, feature_dim=512, num_heads=4):
        super(MTUNet2, self).__init__()
        
        # Complex CNN Encoder shared by both query and support
        self.encoder = CNNEncoder(in_channels, base_features)
        
        # Complex Attention mechanism
        self.attn_module = AttentionModule(feature_dim, num_heads=num_heads)
        
        # Classification Decoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_features*16*8*8, 1024),  # Updated linear layer input size for complex encoder
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, query, support):
        # Step 1: Extract features from the query image using the updated CNNEncoder
        query_features = self.encoder(query)  # Query features are [B, 1024, 8, 8] based on complex CNNEncoder
        
        # Step 2: Extract and aggregate features from the support set
        N = support.size(0)  # Number of support images
        support_features = []
        for i in range(N):
            support_feature = self.encoder(support[i].unsqueeze(0))  # Each support image's features
            support_features.append(support_feature)
        
        # Aggregate support features (using average pooling for simplicity)
        support_features = torch.mean(torch.stack(support_features), dim=0)  # [B, 1024, 8, 8]
        
        # Step 3: Apply complex attention to both query and support features
        query_attn = self.attn_module(query_features)  # Attention on query
        support_attn = self.attn_module(support_features)  # Attention on support
        
        # Step 4: Combine query and support features via one-to-one concatenation
        combined_features = torch.cat((query_attn, support_attn), dim=1)  # Concatenate along the channel dimension
        # Combined features will be [B, 1024 + 1024 = 2048, 8, 8]
        
        # Step 5: Classification Decoder (use the combined query-support features)
        classification_output = self.classifier(combined_features)
        
        return classification_output

import torch
import torch.nn as nn
import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = MTUNet2(in_channels=3, base_features=64, num_classes=5)
criterion_cls = nn.CrossEntropyLoss()  # For classification output
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion_cls, optimizer, epoch):
    model.train()
    running_loss = 0.0
    
    for data, target in enumerate(train_loader):
        
        # Clear gradients
        optimizer.zero_grad()

        # Creating support set
        support = create_support_set(generator, model_EfficientNetV2, model_ShuffleNetV2, target, noise_dim=128)

        # Forward pass
        classification_output = model(data, support)  # Assuming same data for support set in FSL
        
        # Compute loss
        loss_cls = criterion_cls(classification_output, target)  # Assuming target is for classification
        
        # Backward pass
        loss_cls.backward()
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss_cls.item()

        # Compute accuracy for classification output
        _, predicted = torch.max(classification_output.data, 1)
        total += target.size(0)
        correct_cls += (predicted == target).sum().item()

    accuracy = 100 * correct_cls / total
    
    return running_loss / len(train_loader), accuracy


# Evaluation function
def evaluate(model, test_loader, criterion_cls):
    model.eval()
    test_loss = 0.0
    correct_cls = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:

            # Forward pass
            classification_output = model(data)
            
            # Compute loss
            loss_cls = criterion_cls(classification_output, target)
            
            test_loss += loss_cls.item()

            # Compute accuracy for classification output
            _, predicted = torch.max(classification_output.data, 1)
            total += target.size(0)
            correct_cls += (predicted == target).sum().item()

    accuracy = 100 * correct_cls / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy


# Main training loop
num_epochs = 500
for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train(model, train_loader, criterion_cls, optimizer, epoch)
    print(f'Epoch [{epoch}], Training Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    test_loss, test_accuracy = evaluate(model, test_loader, criterion_cls)
    print(f'Epoch [{epoch}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print()
