import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self._train = train
        self._dataset = datasets.cifar.CIFAR100(
            'data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

        self.set_classes_range(0, 10)


    @property
    def _targets(self):
        return np.array(
            self._dataset.train_labels if self._train else self._dataset.test_labels
        )

    def set_classes_range(self, low_range, high_range):
        self._low_range = low_range
        self._high_range = high_range

        if low_range != high_range:
            idxes = np.where(np.logical_and(
                self._targets >= low_range,
                self._targets < high_range
            ))[0]
        else:
            idxes = np.where(self._targets == low_range)[0]

        self._mapping = {
            fake_idx: real_idx
            for fake_idx, real_idx in enumerate(idxes)
        }

    def set_examplars(self, idxes):
        self._mapping.update({
            fake_idx: real_idx
            for fake_idx, real_idx in zip(range(len(self._mapping), len(idxes)), idxes)
        })


    def get_true_index(self, fake_idx):
        return self._mapping[fake_idx]

    def __len__(self):
        return len(self._mapping)

    def __getitem__(self, idx):
        real_idx = self._mapping[idx]
        return self._dataset[real_idx]

