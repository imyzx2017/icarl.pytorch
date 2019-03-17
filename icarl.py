import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

import resnet
import utils
import params


class ICarl(nn.Module):
    def __init__(self, resnet_type="34", n_classes=10, k=2000):
        super().__init__()

        self._k = k
        self._n_classes = n_classes

        self._features_extractor = _get_resnet(resnet_type)
        self._classifier = nn.Linear(self._features_extractor.out_dim, n_classes)
        torch.nn.init.xavier_uniform_(self._classifier.weight)
        self._classifier.bias.data.fill_(0.01)

        self._examplars = {}
        self._means = None

    def forward(self, x):
        x = self._features_extractor(x)
        x = self._classifier(x)
        return x

    def classify(self, images):
        assert self._means is not None
        assert self._means.shape[0] == self._n_classes

        features = self._features_extractor(images)
        features = self._l2_normalize(features)
        return self._get_closest(self._means, features)

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def add_n_classes(self, n):
        self._n_classes += n

        weight = self._classifier.weight.data
        bias = self._classifier.bias.data

        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes)
        torch.nn.init.xavier_uniform_(self._classifier.weight)
        self._classifier.bias.data.fill_(0.01)

        self._classifier.weight.data[:self._n_classes-n] = weight
        self._classifier.bias.data[:self._n_classes-n] = bias

    def _extract_features(self, loader):
        mean = torch.zeros((self._features_extractor.out_dim,))
        features = []
        c = 0

        for inputs, _ in loader:
            inputs = inputs.to(params.DEVICE)
            features.append(self._features_extractor(inputs))
            mean += features[-1].sum(0)
            c += features[-1].shape[0]

        mean /= c
        features = torch.cat(features)

        return self._l2_normalize(features), self._l2_normalize(mean)

    @staticmethod
    def _remove_row(matrix, row_idx):
        return torch.cat((
            matrix[:row_idx, ...],
            matrix[row_idx + 1:, ...]
        ))

    @staticmethod
    def _l2_normalize(tensor):
        return tensor / torch.norm(tensor, p=2)

    @staticmethod
    def _get_closest(centers, features):
        pred_labels = []

        for feature in features:
            distances = torch.pow(centers - feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _get_closest_features(center, features):
        normalized_features = features / torch.norm(features, p=2)
        distances = torch.pow(center - normalized_features, 2).sum(-1)
        return distances.argmin().item()

    def _expand_means(self, class_idx, mean):
        if self._means is None:
            assert class_idx == 0
            self._means = mean[None, ...]
        else:
            assert self._means.shape[0] == class_idx, (self._means.shape, class_idx)
            self._means = torch.cat((self._means, mean[None, ...]))

    @utils.timer
    def build_examplars(self, loader, low_range, high_range):
        examplars = []

        self.eval()
        for class_idx in range(low_range, high_range):
            loader.dataset.set_classes_range(class_idx, class_idx)

            features, class_mean = self._extract_features(loader)
            self._expand_means(class_idx, class_mean)
            examplars_mean = torch.zeros((self._features_extractor.out_dim,))

            for _ in range(min(self._m, features.shape[0])):
                idx = self._get_closest_features(class_mean, features + examplars_mean)
                examplars.append(loader.dataset.get_true_index(idx))
                examplars_mean += features[idx]
                features = self._remove_row(features, idx)

            self._examplars[class_idx] = examplars

        self.train()

    @property
    def examplars(self):
        return np.array([
            examplar_idx
            for class_examplars in self._examplars.values()
            for examplar_idx in class_examplars
        ])

    @utils.timer
    def reduce_examplars(self):
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][:self._m]


def _get_resnet(resnet_type="18"):
    if resnet_type == "18":
        return resnet.resnet18()
    elif resnet_type == "34":
        return resnet.resnet34()
    else:
        raise ValueError("TODO")
