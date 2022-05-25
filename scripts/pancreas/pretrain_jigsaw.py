from argparse import ArgumentParser
import random
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from medical_ssl_3d.data.pancreas import PancreasDataset
from medical_ssl_3d.nn import Encoder3d, ConvBlock3d, Normalize
from medical_ssl_3d.functional import (
    Transform2D, pixelwise_loss
)


class UnsupervisedUNet3d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.backbone = Encoder3d(
            in_channels=1,
            encoder_channels=[16, 32, 64, 128, 256, 512],
            residual=hparams.residual
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1, -1),
            nn.Dropout(p=0.1)
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 1000)
        )
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

    def forward(self, x):
        return self.head(self.backbone(x))

    def _compute_loss(self, batch):
        images, labels = batch
        output = self.forward(images)

        loss = nn.CrossEntropyLoss()
        clf_loss = loss(output, labels)

        return clf_loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        loss = self._compute_loss(batch)
        self.manual_backward(loss)
        self.log('train/cross_entropy', loss, on_step=True, on_epoch=True)

        optimizer.step()

        images, labels = batch
        output = self.forward(images)
        pred_labels = torch.argmax(output, dim=1)

        acc = torch.mean(pred_labels == labels)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


def main(args):
    pl.seed_everything(42, workers=True)
    data_module = PancreasDataset(
        data_dir='/shared/data/pancreas_tumor/Task07_Pancreas',
        cache_dir='/shared/projects/pixelwise-ssl/cache/pancreas',
        train_size=1.0,
        batch_size=args.batch_size,
        num_images_per_epoch=args.num_images_per_epoch,
        num_workers=args.num_workers,
        return_masks=False,
        purpose='jigsaw'
    )
    model = UnsupervisedUNet3d(args)
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
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--out_channels', default=8, type=int)
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=None)
    args = parser.parse_args()

    main(args)