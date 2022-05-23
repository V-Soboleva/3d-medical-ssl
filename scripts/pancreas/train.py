from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from medical_ssl_3d.data.pancreas import PancreasDataset
from medical_ssl_3d.nn import UNet3d, ConvBlock3d
from medical_ssl_3d.functional import compute_dice_loss


class SupervisedUNet3d(pl.LightningModule):
    def __init__(self, pretrained_unet_path=None, residual=False):
        super().__init__()

        self.backbone = UNet3d(
            in_channels=1,
            encoder_channels=[16, 32, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 16],
            residual=residual
        )
        if pretrained_unet_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_unet_path))

        self.head = nn.Sequential(
            ConvBlock3d(in_channels=16, out_channels=8, kernel_size=1),
            ConvBlock3d(in_channels=8, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def _compute_losses(self, batch):
        images, masks = batch
        logits = self.forward(images)
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        dice_loss_pancreas = compute_dice_loss(torch.sigmoid(logits)[:, 0, ...], masks[:, 0, ...], spatial_dims=(-3, -2, -1))
        dice_loss_tumor = compute_dice_loss(torch.sigmoid(logits)[:, 1, ...], masks[:, 1, ...], spatial_dims=(-3, -2, -1))
        
        return bce, dice_loss_pancreas, dice_loss_tumor

    def training_step(self, batch, batch_idx):
        bce, dice_loss_pancreas, dice_loss_tumor = self._compute_losses(batch)
        loss = bce + dice_loss_pancreas + dice_loss_tumor

        self.log('train/train_bce', bce, on_epoch=True)
        self.log('train/train_dice_loss_pancreas', dice_loss_pancreas, on_epoch=True)
        self.log('train/train_dice_loss_tumor', dice_loss_tumor, on_epoch=True)
        self.log('train/train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        bce, dice_loss_pancreas, dice_loss_tumor = self._compute_losses(batch)
        loss = bce + dice_loss_pancreas + dice_loss_tumor

        self.log('val/val_bce', bce, on_epoch=True)
        self.log('val/val_dice_loss_pancreas', dice_loss_pancreas, on_epoch=True)
        self.log('val/val_dice_loss_tumor', dice_loss_tumor, on_epoch=True)
        self.log('val/val_dice_avg', (dice_loss_pancreas + dice_loss_tumor) / 2, on_epoch=True)
        self.log('val/val_loss', loss, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        return torch.sigmoid(self.forward(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=30, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/val_loss',
                'frequency': 30,
            }
        }
        

def main(args):
    pl.seed_everything(42, workers=True)
    data_module = PancreasDataset(
        data_dir='/shared/data/pancreas_tumor/Task07_Pancreas',
        cache_dir='/shared/projects/pixelwise-ssl/cache/pancreas',
        train_size=args.train_size,
        batch_size=args.batch_size,
        num_images_per_epoch=args.num_images_per_epoch,
        num_workers=args.num_workers,
        return_masks=True
    )
    model = SupervisedUNet3d(args.pretrained, residual=args.residual)
    logger = TensorBoardLogger('tb_logs', name=args.name)
    callbacks = [EarlyStopping(monitor='val/val_loss', patience=60)]
    if args.pretrained is not None:
        callbacks.append(BackboneFinetuning(unfreeze_backbone_at_epoch=80, backbone_initial_ratio_lr=1e-3))
    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=1000,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--train_size', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_images_per_epoch', default=250, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=None)
    args = parser.parse_args()

    main(args)
