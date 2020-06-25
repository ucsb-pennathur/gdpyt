import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GdpytInceptionDataset(Dataset):

    __name__ = 'GdpytInceptionDataset'

    def __init__(self, transforms=None, aux_logits=None, normalize_dataset=True, normalize_per_sample=False):
        if transforms is None:
            logger.warning("No transforms specified. Consider specifying at least the transformation ToTensor")
        else:
            self.transforms = transforms

        if normalize_dataset and normalize_per_sample:
            logger.info("Normalize dataset and normalize per sample are both set to true. Normalizing per sample...")
            self.normalize_dataset = False
        else:
            self.normalize_dataset = normalize_dataset
        self.normalize_per_sample = normalize_per_sample

        if aux_logits is not None:
            assert isinstance(aux_logits, list) or isinstance(aux_logits, np.ndarray)
            if not all(np.diff(aux_logits) > 0):
                raise ValueError("Auxiliary logits classes must be defined with a strictly increasing vector")
            self.n_aux_classes = len(aux_logits) + 1

        self.aux_class_encoding = aux_logits

    def __getitem__(self, item):
        source_particle = self._source[item]
        # Use raw image for neural net
        source_particle.use_raw(True)
        target = source_particle.z
        image = source_particle.get_template(resize=self._shape)

        image = Image.fromarray(image.copy(), mode='I;16')

        # Target as a class where each class is an interval
        if self.aux_class_encoding is not None:
            aux_target = self._one_hot_class(target)
        else:
            aux_target = None

        # Normal target
        target = np.array([target])

        if self.transforms:
            image = self.transforms(image)

        # Float transform for mean calculation
        image = image.float()

        if self.normalize_per_sample:
            image = (image - image.mean()) / image.std()

        if self._mode == 'train':
            if aux_target is None:
                sample = {'input': image, 'target': torch.from_numpy(target)}
            else:
                sample = {'input': image, 'target': torch.from_numpy(target),
                          'aux_target': torch.from_numpy(aux_target).long()}
        else:
            sample = {'input': image}

        return sample

    def __len__(self):
        return len(self._source)

    def _compute_stats(self):
        self.stats = {'mean': 0, 'std': 1}
        inputs = []
        for idx in range(len(self)):
            x = self.__getitem__(idx)['input']
            inputs.append(x)
        all_inputs = torch.cat(inputs, 0)
        self.stats = {'mean': all_inputs.mean().item(),
                      'std': all_inputs.std().item()}
        logger.info("Computed statistics. \n"
                    "Mean: {}\n"
                    "Std: {}".format(self.stats['mean'], self.stats['std']))
        if self.normalize_dataset and self._mode == 'train':
            logger.info("Setting computed statistics as normalization parameters")
            self.transforms = Compose([self.transforms, Normalize(**self.stats)])

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

    def _one_hot_class(self, y):
        n_cls = self.n_aux_classes
        if all(y < self.aux_class_encoding):
            cls = 0
        elif all(y > self.aux_class_encoding):
            cls = n_cls - 1
        else:
            cls = min(np.argwhere(y < self.aux_class_encoding))
        ret = np.zeros(n_cls)
        ret[cls] = 1

        return ret

    def from_calib_set(self, calib_set, max_size=None, skip_na=True, min_stack_len=10):
        # Identify largest template in calibration set
        all_ = []
        templ_dim = 0
        skip_stacks = []
        for stack_id, stack in calib_set.calibration_stacks.items():
            if max_size is not None:
                if max(stack.shape) > max_size:
                    skip_stacks.append(stack_id)
                    continue
            if len(stack) < min_stack_len:
                skip_stacks.append(stack_id)
                continue
            if max(stack.shape) > templ_dim:
                templ_dim = max(stack.shape)

        logger.info("Max. size specified: {}. Shape of calibration set: {}".format(max_size, (templ_dim, templ_dim)))
        self._shape = (templ_dim, templ_dim)

        # Load all calibration stacks in this calibration set
        for stack_id, stack in calib_set.calibration_stacks.items():
            if stack_id not in skip_stacks:
                all_ += self._load_calib_stack(stack, skip_na=skip_na)
        self._source = all_
        logger.info("Created a {} as a training set using {} particles from calibration set".format(self.__name__, len(all_)))

        # When loading from calibration set or stack it's always a training set
        self._mode = 'train'
        self._compute_stats()

    def from_image_collections(self, collection, template_shape=None, max_size=None, skip_na=True):

        all_ = []

        if template_shape is None:
            templ_dim = 0
            for image in collection.images.values():
                for particle in image.particles:
                    # Raw templates are used in neural net
                    if max_size is not None:
                        if max(particle.bbox[2:]) > max_size:
                            continue
                    if max(particle.bbox[2:]) > templ_dim:
                        templ_dim = max(particle.bbox[2:])
            self._shape = (templ_dim, templ_dim)
        else:
            assert isinstance(template_shape, tuple)
            self._shape = template_shape

        for image in collection.images.values():
            for particle in image.particles:
                particle.use_raw(True)
                if max(particle.bbox[2:]) > max_size:
                    continue
                template = particle.get_template(resize=self.shape)
                if skip_na and np.isnan(template).sum() != 0:
                    continue
                all_.append(particle)

        self._source = all_

        if collection.is_infered():
            logger.info(
            "Created a {} as a test set with shape {} using {} particles from "
            "GdpytImageCollection in {}".format(self.__name__, self.shape, len(all_), collection.folder))
            self._mode = 'train'
        else:
            logger.info(
                "Created a {} as a prediction set (unknown targets) with shape {} using {} particles from "
                "GdpytImageCollection in {}".format(self.__name__, self.shape, len(all_), collection.folder))
            self._mode = 'eval'
        self._compute_stats()

    def infer(self, model, idx, device=None, batch_size=64):
        """
        Infer a sample in the dataset with a trained model
        """

        if device is None:
            device = torch.device('cpu')

        if idx is not None:
            x = self.__getitem__(idx)['input'].float().to(device)
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
            # Evaluation mode
            model.eval()

            ys = []
            infer_idx = [()]
            for j in range(0, len(self), batch_size):
                end_idx = j + batch_size
                if end_idx > len(self):
                    end_idx = len(self)
                inputs = [self.__getitem__(i)['input'].float().unsqueeze_(0) for i in range(j, end_idx)]
                inputs = torch.cat(inputs, 0).to(device)

                y = model(inputs)
                ys.append(y)

            y = torch.cat(ys, 0)

            if self._mode in ['train', 'test']:
                targets = [self.__getitem__(i)['target'] for i in range(len(self))]
                return y, targets
            else:
                return y

    def set_sample_z(self, idx, z):
        if isinstance(z, torch.Tensor):
            if len(z) == 1:
                z = z.item()
                self._source[idx].set_z(z)
            else:
                for idx, z_ in enumerate(z):
                    self._source[idx].set_z(z_.item())

    @property
    def shape(self):
        return self._shape

    @property
    def mode(self):
        return self._mode