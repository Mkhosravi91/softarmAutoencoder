# Model and DataModule instantiation
import os
from Autoencode import AutoEncoder, CustomImageDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import glob
import torch
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as transforms
from Autoencode import CustomImageDataset
import pdb
from pytorch_lightning.callbacks import EarlyStopping

####for training
# model = AutoEncoder()
###### 
data_module = CustomImageDataModule(data_dir='/data/Mahsa/Br2/imagedatapair')
#data_module_test = CustomImageDataModule(data_dir='/data/Mahsa/br2sim/testB')
transform = transforms.Compose([
             transforms.Resize((640, 480)),
             transforms.ToTensor(),])
# dataset = CustomImageDataset('/data/Mahsa/br2sim/trainB', transform=transform)
# dataloader = DataLoader(data_module,batch_size = 16)
wandb_logger = WandbLogger(project='Autoencoder')
#####for fine tunning
best_model_path ='/data/Mahsa/Br2/my_model/autoencoder-epoch=00-val_loss=1.42_thirdrun.ckpt'
model = AutoEncoder.load_from_checkpoint(best_model_path)
#####
# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my_model/',
    filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)
# early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=150, verbose=True, mode="min")

# Trainer
trainer = pl.Trainer(
    max_epochs=300,
    callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None, 
    accumulate_grad_batches=8,
    logger=wandb_logger
)

# torch.cuda.empty_cache()
# # Train
data_module.setup(stage='fit')
trainer.fit(model,data_module.train_dataloader(), data_module.val_dataloader())

# # Test
#data_module_test.setup(stage='test')
# trainer.test(model, dataloaders=data_module.test_dataloader())
# pdb.set_trace()
# Load the best model from checkpoint
# best_model_path = checkpoint_callback.best_model_path
#best_model_path ='/data/Mahsa/br2sim/my_model/autoencoder-epoch=48-val_loss=2.30.ckpt'
#model = AutoEncoder.load_from_checkpoint(best_model_path)

# trainer.test(model, dataloaders=dataloader)
