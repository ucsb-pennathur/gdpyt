import numpy as np
import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        outp_sample = {}
        image = sample['input']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        outp_sample.update({'input': torch.from_numpy(image).float()})

        if 'target' in sample.keys():
            target = sample['target']
            outp_sample.update({'target': torch.from_numpy(target).float()})

        return outp_sample

class RotateN90(object):
    """ Rotate an image by a multiple of 90 degrees"""

    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n

    def __call__(self, sample):
        image = sample['input']
        sample.update({'input': np.rot90(image, k=self.n, axes=(0,1))})

        return sample