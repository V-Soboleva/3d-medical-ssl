from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dpipe.im.patch import get_random_box

from medical_ssl_3d.data.pancreas import PancreasDataset
from medical_ssl_3d.nn import UNet3d, ConvBlock3d
from medical_ssl_3d.functional import compute_dice_loss


class UnsupervisedUNet3d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.backbone = UNet3d(
            in_channels=1,
            encoder_channels=[16, 32, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 16],
            residual=hparams.residual
        )

        self.head = nn.Sequential(
            ConvBlock3d(in_channels=16, out_channels=8, kernel_size=1),
            ConvBlock3d(in_channels=8, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def _compute_losses(self, batch):
        images = batch
        patch_size = (40, 40, 40)
        lam = 0.5

        box1  = get_random_box(images[0][0].shape, patch_size)
        box2  = get_random_box(images[1][0].shape, patch_size)

        masks = torch.zeros_like(images) > 0
        masks[0, :, box1[0][0]: box1[1][0], box1[0][1]: box1[1][1], box1[0][2]: box1[1][2]] = torch.randn(size=patch_size) > 0.3
        masks[1, :, box2[0][0]: box2[1][0], box2[0][1]: box2[1][1], box2[0][2]: box2[1][2]] = torch.randn(size=patch_size) > 0.3

        roi = images > 0
        masks[~roi] = 0

        images[masks] = lam * images[masks] + (1-lam) * torch.flip(images, dims=(0, 1))[masks]

        logits = self.forward(images)
        bce = F.binary_cross_entropy_with_logits(logits, masks.type(dtype=torch.float))
        dice_loss = compute_dice_loss(torch.sigmoid(logits), masks, spatial_dims=(-3, -2, -1))
        
        return bce, dice_loss

    def training_step(self, batch, batch_idx):
        bce, dice_loss = self._compute_losses(batch)
        loss = bce + dice_loss

        self.log('train/train_bce', bce, on_epoch=True)
        self.log('train/train_dice_loss', dice_loss, on_epoch=True)
        self.log('train/train_loss', loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
        

def main(args):
    pl.seed_everything(42, workers=True)
    data_module = PancreasDataset(
        data_dir='/shared/data/pancreas_tumor/Task07_Pancreas',
        cache_dir='/shared/projects/pixelwise-ssl/cache/pancreas1',
        train_size=1.0,
        batch_size=2,
        num_images_per_epoch=args.num_images_per_epoch,
        num_workers=args.num_workers,
        return_masks=False
    )
    model = UnsupervisedUNet3d(args)
    logger = TensorBoardLogger('tb_logs_new', name=args.name)
    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=200
    )
    trainer.fit(model, datamodule=data_module)
    torch.save(model.backbone.state_dict(), f'{logger.log_dir}/backbone.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--num_images_per_epoch', default=150, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=None)
    args = parser.parse_args()

    main(args)