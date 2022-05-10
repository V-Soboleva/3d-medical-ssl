from pathlib import Path
import numpy as np
import nibabel
from connectome import Source, Output, meta

MODE_2_FOLDER = {'train' : 'imagesTr', 'test' : 'imagesTs'}
ALL_LABELS = [1,2]
TUMOR_LABEL = 2


class Pancreas(Source):
    """Interface for loading Pancreas data based on connectome (https://github.com/neuro-ml/connectome).

    Args:
    data_root (Union[str, os.PathLike]): path to a directory "Task07_Pancreas"
    mode (str): train or test

    Examples:
        >>> dataset = Pancreas(data_dir='/path/to/data', mode='train')
        >>> id_ = dataset.ids[0]
        >>> image = dataset.image(id_)
        >>> spacing_mm = dataset.spacing_mm(id_)
        >>> mask = dataset.mask(id_)
    """
    _data_dir: str
    _mode: str = 'train'

    def _root(_data_dir):
        return Path(_data_dir)

    @meta
    def ids(_root, _mode):
        path = _root / MODE_2_FOLDER[_mode]
        return sorted([file.name for file in path.iterdir() if not file.name.startswith('.')])

    def _image(id_, _root, _mode):
        return nibabel.load(_root / MODE_2_FOLDER[_mode] / id_)
    
    def image(_image):
        return _image.get_fdata()

    def spacing_mm(_image):
        return _image.header.get_zooms()

    def seg(id_, _root, _mode):
        return nibabel.load(_root / 'labelsTr' / id_) if _mode == 'train' else None

    def mask(seg: Output, _mode):
        if _mode == 'test':
            return None
        else:
            labels = seg.get_fdata()
            return np.stack([
                np.isin(labels, ALL_LABELS),
                labels == TUMOR_LABEL
            ])
