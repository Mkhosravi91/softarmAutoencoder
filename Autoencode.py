import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import wandb
import glob
import pdb
import torchvision.transforms.functional as TF


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_dir, '*.png'))  # assuming images are in JPG format
        # pdb.set_trace()
        print('here')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # returning dummy label for compatibility

# class CustomImageDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir='./data', batch_size=16):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.transform = transforms.Compose([
#             transforms.Resize((640, 480)),
#             transforms.ToTensor(),
#         ])
#         self.train = None
#         self.val = None
#         self.test = None

    # def setup(self, stage=None):
    #     dataset = CustomImageDataset(self.data_dir, transform=self.transform)
    #     total_size = len(dataset)
    #     train_size = int(0.8 * total_size)
    #     val_size = total_size - train_size
    #     # pdb.set_trace()
    #     if stage == 'fit' or stage is None:
    #         self.train, self.val = random_split(dataset, [train_size, val_size])
    #     if stage == 'test' or stage is None:
    #         self.test = CustomImageDataset(self.data_dir, transform=self.transform)

    # def train_dataloader(self):
    #     return DataLoader(self.train, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.val, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size)
    
class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        resnet = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # self.compressor = nn.Sequential(nn.Flatten(),
        #                                 nn.Linear(153600,15360),
        #                                 nn.Linear(15360,1536),
        #                                 nn.Linear(1536,256)
        # )
        
        # self.decompressor = nn.Sequential(nn.Linear(256,1536),
        #                                 nn.Linear(1536,15360),
        #                                 nn.Linear(15360,153600),
        #                                 nn.Unflatten(1, (512, 20, 15))
        #                                 )
        self.compressor = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Output: (512, 10, 8)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=3, padding=1),  # Output: (512, 4, 3)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 3, 256)
        )

        self.decompressor = nn.Sequential(
            nn.Linear(256, 512 * 20 * 15),
            nn.ReLU(),
            nn.Unflatten(1, (512, 20, 15))
        )
        
        # self.decompressor = nn.Sequential(
        #     nn.Linear(256, 512 * 4 * 3),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (512, 4, 3)),
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=3, padding=1, output_padding=0),  # Output: (512, 10, 8)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (512, 20, 15)
        #     nn.ReLU()
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        # print(f'Encoded shape: {encoded.shape}')
        compressed = self.compressor(encoded)
        # print(f'Compressed shape: {compressed.shape}')
        decompressed = self.decompressor(compressed)
        # print(f'Decompressed shape: {decompressed.shape}')
        decoded = self.decoder(decompressed)
        # print(f'Decoded shape: {decoded.shape}') 
        return decoded

    def training_step(self, batch):
        x, _ = batch
        loss = self.compute_losses(x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        loss = self.compute_losses(x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, _ = batch
        # pdb.set_trace()
        loss = self.compute_losses(x)
        self.log('test_loss', loss)
        return loss

    # def compute_losses(self, x):
    #     loss = 0
    #     loss += self.inpainting_loss(x)
    #     loss += self.colorization_loss(x)
    #     loss += self.denoising_loss(x)
    #     loss += self.context_prediction_loss(x)
    #     loss += self.rotation_prediction_loss(x)
    #     loss += self.contrastive_learning_loss(x)
    #     return loss
# more focused reconstructed and masked loss
    def compute_losses(self, x):
        # Reconstruct the image
        x_reconstructed = self(x)
        
        # Create masks for red ball and green tip
        red_ball_mask = (x[:, 0, :, :] > 0.6) & (x[:, 1, :, :] < 0.3) & (x[:, 2, :, :] < 0.3)  # Adjust thresholds as needed
        green_tip_mask = (x[:, 1, :, :] > 0.6) & (x[:, 0, :, :] < 0.3) & (x[:, 2, :, :] < 0.3)  # Adjust thresholds as needed
        
        # Combine the masks
        mask = (red_ball_mask | green_tip_mask).float()
        
        # Compute masked loss
        masked_loss = self.masked_mse_loss(x_reconstructed, x, mask)
        self.log('masked_loss', masked_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute other losses
        inpainting_loss = self.inpainting_loss(x) * 0.1
        self.log('inpainting_loss', inpainting_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        denoising_loss = self.denoising_loss(x) * 0.1
        self.log('denoising_loss', denoising_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Combine the losses
        total_loss = masked_loss + inpainting_loss + denoising_loss

        # Log the total loss
        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss


    def inpainting_loss(self, x):
        mask = torch.rand_like(x) > 0.8
        x_masked = x.clone()
        x_masked[mask] = 0
        x_reconstructed = self(x_masked)
        # pdb.set_trace()
        return F.mse_loss(x_reconstructed[mask], x[mask])

    def colorization_loss(self, x):
        grayscale = transforms.Grayscale()(x)
        grayscale = grayscale.repeat(1, 3, 1, 1)
        x_reconstructed = self(grayscale)
        return F.mse_loss(x_reconstructed, x)

    def denoising_loss(self, x):
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise
        x_reconstructed = self(x_noisy)
        return F.mse_loss(x_reconstructed, x)

    # def context_prediction_loss(self, x):
    #     patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
    #     patches = patches.contiguous().view(x.size(0), -1, 3, 16, 16)
    #     idx = torch.randperm(patches.size(1))
    #     patches_shuffled = patches[:, idx, :, :, :]
    #     x_shuffled = patches_shuffled.view(x.size(0), 3, 32, 32)
    #     x_reconstructed = self(x_shuffled)
    #     return F.mse_loss(x_reconstructed, x)
    def context_prediction_loss(self, x):
        # Assuming input size (B, C, H, W) with B=32, C=3, H=640, W=480
        B, C, H, W = x.size()
        patch_size = 16  # Example patch size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, num_patches_h * num_patches_w, -1)

        idx = torch.randperm(patches.size(1))
        patches_shuffled = patches[:, idx, :]
        patches_shuffled = patches_shuffled.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        patches_shuffled = patches_shuffled.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        x_reconstructed = self(patches_shuffled)
        return F.mse_loss(x_reconstructed, x)

    def rotation_prediction_loss(self, x):
        angles = [0, 90, 180, 270]
        angle = angles[torch.randint(0, 4, (1,)).item()]
        x_rotated = transforms.functional.rotate(x, angle)
        x_reconstructed = self(x_rotated)
        target_angle = torch.tensor([angle // 90] * x.size(0), dtype=torch.long, device=self.device)
        pred_angle = self.encoder(x_reconstructed).mean(dim=[2, 3])
        return F.cross_entropy(pred_angle, target_angle)

    def contrastive_learning_loss(self, x):
            resize = transforms.Resize((224, 224))
            x_resized = resize(x)

            aug1 = TF.hflip(x_resized)
            aug2 = TF.vflip(x_resized)

            h1 = self.encoder(aug1).view(aug1.size(0), -1)
            h2 = self.encoder(aug2).view(aug2.size(0), -1)

            return F.cosine_embedding_loss(h1, h2, torch.ones(h1.size(0), device=self.device))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer




class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_dir, '*.png'))  # assuming images are in JPG format

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # returning dummy label for compatibility

class CustomImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((640, 480)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = CustomImageDataset(self.data_dir, transform=self.transform)
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            self.train, self.val = random_split(dataset, [train_size, val_size])  # 80-20 split
        if stage == 'test' or stage is None:
            self.test = CustomImageDataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
