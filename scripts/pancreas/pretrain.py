from argparse import ArgumentParser
import random
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from medical_ssl_3d.data.pancreas import PancreasDataset
from medical_ssl_3d.nn import UNet3d, ConvBlock3d, Normalize
from medical_ssl_3d.functional import (
    Transform2D, pixelwise_loss
)


class UnsupervisedUNet3d(pl.LightningModule):
    def __init__(self, out_channels, hparams):
        super().__init__()

        self.backbone = UNet3d(
            in_channels=1,
            encoder_channels=[16, 32, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 16],
            residual=hparams.residual
        )
        self.head = nn.Sequential(
            ConvBlock3d(in_channels=16, out_channels=8, kernel_size=1),
            ConvBlock3d(in_channels=8, out_channels=out_channels, kernel_size=1),
            Normalize(dim=1)
        )
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

    def forward(self, x):
        return self.head(self.backbone(x))

    def _compute_pixelwise_loss(self, image, roi):
        hparams = self.hparams
        transform = Transform2D.random(
            # randomly choose plane (vdim, hdim) in which random transformation is applied
            dims=random.choice([(-3, -2), (-3, -1), (-2, -1)]),
            shape=image.shape,
            vflip_p=hparams.vflip_p,
            hflip_p=hparams.hflip_p,
            max_angle=hparams.max_angle,
            max_scale=hparams.max_scale,
            max_shift=hparams.max_shift,
        )
        return pixelwise_loss(image, self, transform, hparams.temperature, hparams.min_neg_distance_vxl, roi)

    def training_step(self, batch, batch_idx):
        image = batch
        roi = torch.any(image.cpu() > 0, dim=0)

        optimizer = self.optimizers()
        optimizer.zero_grad()

        pixelwise_loss = self._compute_pixelwise_loss(image, roi)
        self.manual_backward(pixelwise_loss)
        self.log('loss/pixelwise_loss', pixelwise_loss, on_step=True, on_epoch=True)

        optimizer.step()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


def main(args):
    pl.seed_everything(42, workers=True)
    data_module = PancreasDataset(
        data_dir='/shared/data/pancreas_tumor/Task07_Pancreas',
        cache_dir='/shared/projects/pixelwise-ssl/cache/pancreas',
        train_size=1.0,
        batch_size=None,
        num_images_per_epoch=args.num_images_per_epoch,
        num_workers=args.num_workers,
        return_masks=False
    )
    model = UnsupervisedUNet3d(args.out_channels, args)
    logger = TensorBoardLogger('tb_logs_new', name=args.name)
    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=100
    )
    trainer.fit(model, datamodule=data_module)
    torch.save(model.backbone.state_dict(), f'{logger.log_dir}/backbone.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--num_images_per_epoch', default=300, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--out_channels', default=8, type=int)
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--min_neg_distance_vxl', default=10, type=float)
    parser.add_argument('--vflip_p', default=0, type=float)
    parser.add_argument('--hflip_p', default=0, type=float)
    parser.add_argument('--max_angle', default=30, type=float)
    parser.add_argument('--max_scale', default=1.5, type=float)
    parser.add_argument('--max_shift', default=0.1, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=None)
    args = parser.parse_args()

    main(args)