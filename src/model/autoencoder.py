import torch
import torch.nn as nn
import os

# Define the DoubleConv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)


# Define the Autoencoder model
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.1, size_factor=1, checkpoint=None):
        super(AutoEncoder, self).__init__()

        # Encoder path
        self.enc1 = DoubleConv(in_channels, int(64*size_factor), dropout_rate)
        self.enc2 = DoubleConv(int(64*size_factor), int(128*size_factor), dropout_rate * 1.5)
        self.enc3 = DoubleConv(int(128*size_factor), int(256*size_factor), dropout_rate * 2)

        # Bottleneck
        self.bottleneck = DoubleConv(int(256*size_factor), int(512*size_factor), dropout_rate * 2.5)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(int(512*size_factor), int(256*size_factor), kernel_size=2, stride=2)
        self.dec3 = DoubleConv(int(256*size_factor), int(256*size_factor), dropout_rate * 2)

        self.upconv2 = nn.ConvTranspose2d(int(256*size_factor), int(128*size_factor), kernel_size=2, stride=2)
        self.dec2 = DoubleConv(int(128*size_factor), int(128*size_factor), dropout_rate * 1.5)

        self.upconv1 = nn.ConvTranspose2d(int(128*size_factor), int(64*size_factor), kernel_size=2, stride=2)
        self.dec1 = DoubleConv(int(64*size_factor), int(64*size_factor), dropout_rate)

        # Output layer
        self.out_conv = nn.Conv2d(int(64*size_factor), in_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc3))

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out_conv(dec1)
        out = torch.sigmoid(out)

        return out


class AutoEncodingDataset(Dataset):
    def __init__(self, image_path, label_path):


    def __len__(self):

    def __getitem__(self, idx):


def train_autoencoding(save_path: str,
                       img_path: dict,
                       label_path: dict,):

    os.makedirs(save_path, exist_ok=True)

    # Build the autoencoder
    model = AutoEncoder()
    model = model.to('cuda')

    # Define dataloaders

    train_dataset = AutoEncodingDataset(
        r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\train\images',
        r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\train\labels')

    val_dataset = AutoEncodingDataset(
        r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\val\images',
        r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\val\labels')

    train_dataloader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=1,)

    val_dataloader = DataLoader(val_dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=1, )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * nb_epochs, eta_min=0.00001)

    # Training loop
    for epoch in range(nb_epochs):
        model.train()
        epoch_loss = 0

        for images in tqdm(train_dataloader):

            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(images)

            # Compute loss
            loss = criterion(reconstructed, images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        val_index = 0
        with torch.no_grad():
            for val_images in val_dataloader:
                val_images = val_images.to(device)
                val_reconstructed = model(val_images)
                val_loss += criterion(val_reconstructed, val_images).item()

                # Save the concatenated validation images with index
                concatenated_val = torch.cat((val_images.cpu(), val_reconstructed.cpu()), dim=0)
                val_index += 1

        print(
            f'Epoch [{epoch + 1}/{nb_epochs}], Train Loss: {epoch_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}')


if __name__ == '__main__':
    train_autoencoding()