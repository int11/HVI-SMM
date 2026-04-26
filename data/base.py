import random
import numpy as np
import torch


class PairedTransformMixin:
    """Provides seed-synced transform helpers for paired image datasets.

    Use `_apply_paired_transform` for two images that must share the same
    random seed (e.g. low/high pairs needing identical crop/flip). Use
    `_apply_independent_transform` to transform a single image with a fresh
    seed (used by unpaired datasets where the two images should not share
    augmentations).
    """

    @staticmethod
    def _make_seed():
        seed = random.randint(1, 1000000)
        return int(np.random.randint(seed))

    def _apply_paired_transform(self, im1, im2):
        if not getattr(self, 'transform', None):
            return im1, im2
        seed = self._make_seed()
        random.seed(seed)
        torch.manual_seed(seed)
        im1 = self.transform(im1)
        random.seed(seed)
        torch.manual_seed(seed)
        im2 = self.transform(im2)
        return im1, im2

    def _apply_independent_transform(self, im):
        if not getattr(self, 'transform', None):
            return im
        seed = int(np.random.randint(1000000))
        random.seed(seed)
        torch.manual_seed(seed)
        return self.transform(im)
