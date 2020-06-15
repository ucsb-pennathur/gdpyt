from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gdpyt.utils.plotting import plot_tensor_dset
from .transforms import ToTensor
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GdpytTensorDataset(Dataset):

    __name__ = 'GdpytTensorDataset'

    def __init__(self, transforms_=None, normalize=False, tset_stats=None):

        """
        dataloader class
        """
        self.normalize = normalize
        self._source = None
        sample_transforms = []
        if transforms_ is not None:
            for transf in transforms_:
                sample_transforms.append(transf)

        # Tensor transformation is done in every case
        sample_transforms.append(ToTensor())
        self.transform = transforms.Compose(sample_transforms)
        self._shape = None
        # Statistics, mean and variance
        self.stats = None
        # Statistics from train set. These are applies if this set is a test or prediction set
        self._stats_from_tset = tset_stats
        self._mode = None

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        source_particle = self._source[idx]

        # Use raw image for neural net
        source_particle.use_raw(True)
        target = source_particle.z
        image = source_particle.get_template(resize=self._shape)

        target = np.array([target])

        if self._mode in ['train', 'test']:
            sample = {'input': image, 'target': target}
        else:
            sample = {'input': image}

        if self.transform:
            sample = self.transform(sample)

        if self.normalize:
            image = sample['input']
            image = (image - self.stats['mean']) / self.stats['std']
            sample.update({'input': image})

        return sample

    def _compute_stats(self):
        if self._mode in ['test', 'predict'] and self._stats_from_tset is not None:
            logger.info("Using user-defined normalization statistics (e.g. values from train set): {}".format(self._stats_from_tset))
            assert 'mean' in self._stats_from_tset.keys() and 'std' in self._stats_from_tset.keys()
            self.stats = self._stats_from_tset
        else:
            imgs = []
            self.stats = {'mean': 0, 'std': 1}
            inputs = []
            for idx in range(len(self)):
                x = self.__getitem__(idx)['input']
                inputs.append(x)
            all_inputs = torch.cat(inputs, 0)
            self.stats = {'mean': all_inputs.mean(), 'std': all_inputs.std()}

    def _load_calib_stack(self, stack, skip_na=True):
        all_ = []
        for particle in stack.particles:
            particle.use_raw(True)
            if stack.shape == self.shape:
                template = particle.get_template()
            else:
                template = particle.get_template(resize=self.shape)
            if skip_na and np.isnan(template).sum() != 0:
                continue
            all_.append(particle)
        return all_

    def from_calib_set(self, calib_set, max_size=None, skip_na=True, min_stack_len=10):
        # Identify largest template in calibration set
        w_max, h_max = (0, 0)
        skip_stacks = []
        for stack_id, stack in calib_set.calibration_stacks.items():
            w, h = stack.shape
            if min_stack_len is not None:
                if len(stack) < min_stack_len:
                    skip_stacks.append(stack_id)
                    continue
            if max_size is not None:
                if w > max_size or h > max_size:
                    skip_stacks.append(stack_id)
                    continue
            if w > w_max:
                w_max = w
            if h > h_max:
                h_max = h
        logger.info("Max. size specified: {}. Shape of calibration set: {}".format(max_size, (w_max, h_max)))
        self._shape = (w_max, h_max)

        # Load all calibration stacks in this calibration set
        all_ = []
        for stack_id, stack in calib_set.calibration_stacks.items():
            if stack_id not in skip_stacks:
                all_ += self._load_calib_stack(stack, skip_na=skip_na)
        self._source = all_
        logger.info("Created a {} as a training set using {} particles from calibration set".format(self.__name__, len(all_)))

        # When loading from calibration set or stack it's always a training set
        self._mode = 'train'
        self._compute_stats()

    def from_image_collection(self, collection, ref_shape=None, max_size=None, skip_na=True):
        if ref_shape is None:
            logger.error("A shape as a 2 element tuple when loading a test set from an image collection. "
                         "This should be the shape of a sample from the training set")
            raise TypeError
        else:
            assert isinstance(ref_shape, tuple)
            assert len(ref_shape) == 2
            self._shape = ref_shape

        all_ = []
        for image in collection.images.values():
            for particle in image.particles:
                # Raw templates are used in neural net
                particle.use_raw(True)
                if max_size is not None:
                    w, h = particle.bbox[2:]
                    if w > max_size or h > max_size:
                        continue
                template = particle.get_template(resize=ref_shape)
                if skip_na and np.isnan(template).sum() != 0:
                    continue
                all_.append(particle)

        self._source = all_

        if collection.is_infered():
            logger.info(
            "Created a {} as a test set using {} particles from "
            "GdpytImageCollection in {}".format(self.__name__, len(all_), collection.folder))
            self._mode = 'test'
        else:
            logger.info(
                "Created a {} as a prediction set (unknown targets) using {} particles from "
                "GdpytImageCollection in {}".format(self.__name__, len(all_), collection.folder))
            self._mode = 'predict'
        self._compute_stats()

    def infer(self, model, idx, device=None):
        """
        Infer a sample in the dataset with a trained model
        """

        if device is None:
            device = torch.device('cpu')

        if idx is not None:
            x = self.__getitem__(idx)['input'].to(device)
            # Force mini-batch shape
            x.unsqueeze_(0)

            # Evaluation mode
            model.eval()
            y = model(x)

            if self._mode in ['train', 'test']:
                target = self.__getitem__(idx)['target']
                logger.info("Predicted: {}, Target: {}".format(y.item(), target.item()))
                return y.item(), target.item()
            else:
                logger.info("Predicted: {}".format(y.item()))
                return y.item()
        else:
            inputs = [self.__getitem__(i)['input'].unsqueeze_(0) for i in range(len(self))]
            inputs = torch.cat(inputs, 0).to(device)

            # Evaluation mode
            model.eval()
            y = model(inputs)

            if self._mode in ['train', 'test']:
                targets = [self.__getitem__(i)['target'] for i in range(len(self))]
                return y, targets
            else:
                return y

    def plot(self, N):
        assert isinstance(N, int) and N > 0
        fig = plot_tensor_dset(self, N)
        return fig

    def return_dataloader(self, batch_size=4, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def set_sample_z(self, idx, z):
        if isinstance(z, torch.Tensor):
            if len(z) == 1:
                z = z.item()
                self._source[idx].set_z(z)
            else:
                for idx, z_ in enumerate(z):
                    self._source[idx].set_z(z_.item())

    @property
    def input_shape(self):
        return
    @property
    def shape(self):
        return self._shape

    @property
    def mode(self):
        return self._mode