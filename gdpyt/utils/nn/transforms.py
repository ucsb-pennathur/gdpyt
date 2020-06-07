import numpy as np
import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        outp_sample = {}
        image = sample['input']

        # Add channel dimension if array is only a 2D image
        if len(image.shape) == 2:
            image = np.nan_to_num(image[:, :, np.newaxis])
        else:
            image = np.nan_to_num(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float)
        outp_sample.update({'input': torch.from_numpy(image).float()})

        if 'target' in sample.keys():
            target = sample['target']
            outp_sample.update({'target': torch.from_numpy(target).float()})

        return outp_sample

class RotateN90(object):
    """ Rotate an image by a random multiple of 90 degrees"""

    def __call__(self, sample):
        image = sample['input']
        n = np.random.randint(0, 4)
        sample.update({'input': np.rot90(image, k=0, axes=(0, 1))})

        return sample

class RandomBCGAdjust(object):
    """ Random brightness and a contrast adjustment"""

    def __call__(self, sample):
        image = sample['input']
        img_range = (image.max() - image.min()).astype(np.int64)
        br = np.random.randint(- img_range / 2, high=img_range / 2)

        # Random contrast adjustment between 0.5 and 1.5
        contr = np.random.uniform(0.5, 1.5)
        sample.update({'input': np.clip((image * contr) + br, 0, None)})

        return sample

