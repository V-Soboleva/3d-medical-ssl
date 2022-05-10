from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, ids, load_func):
        self.ids = ids
        self.load_func = load_func

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.load_func(self.ids[idx])
