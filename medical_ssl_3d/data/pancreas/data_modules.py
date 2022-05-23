from pathlib import Path
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from connectome import Chain, Transform, Apply, CacheToDisk

from .sources import Pancreas
from ..processing import CropToBox, ZoomToShape, scale_hu, min_max_scale
from ..utils import mask_to_box
from ..split import kfold
from ..torch import TorchDataset

IMAGE_SIZE = 128, 128, 128


class PancreasDataset(pl.LightningDataModule):
    def __init__(self, data_dir, cache_dir, train_size, num_images_per_epoch, 
                return_masks=True, random_state=42, batch_size=1, num_workers=0):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.train_size = train_size
        self.num_images_per_epoch = num_images_per_epoch
        self.return_masks = return_masks
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        preprocessed = Chain(
            Pancreas(data_dir=self.data_dir),
            Apply(image=lambda x: np.expand_dims(x, axis=0)),
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_box(np.any(image > 0, axis=0))),
            CropToBox(axis=(-3, -2, -1)), 
            ZoomToShape(shape=IMAGE_SIZE, axis=(-3, -2, -1)),
            Apply(image=partial(scale_hu, window_hu=(-200, 300))),
            Apply(image=np.float16),
            CacheToDisk.simple('image', 'mask', root=self.cache_dir),
            Apply(image=np.float32, mask=np.float32),
        )

        train_ids, val_ids = train_test_split(preprocessed.ids, test_size=0.3, shuffle=True, random_state=self.random_state)
        train_size = int(len(train_ids) * self.train_size)
        train_ids = train_ids[:train_size]

        if self.return_masks:
            self.train_dataset = TorchDataset(train_ids, preprocessed._compile(['image', 'mask']))
            self.val_dataset = TorchDataset(val_ids, preprocessed._compile(['image', 'mask']))
        else:
            self.train_dataset = TorchDataset(train_ids, preprocessed._compile('image'))
            self.val_dataset = TorchDataset(val_ids, preprocessed._compile('image'))

        self.test_dataset = TorchDataset(val_ids, preprocessed._compile('image'))

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, num_samples=self.num_images_per_epoch)
        return DataLoader(self.train_dataset, self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers)