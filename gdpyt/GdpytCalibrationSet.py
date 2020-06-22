from .GdpytCalibrationStack import GdpytCalibrationStack
from gdpyt.similarity.nn import GdpytNet, train_net
from gdpyt.inception_net import GdpytInceptionDataset, GdpytInceptionRegressionNet
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from math import sqrt
import logging

logger = logging.getLogger(__name__)

class GdpytCalibrationSet(object):

    def __init__(self, collections, image_to_z, exclude=[], dilate=True):
        super(GdpytCalibrationSet, self).__init__()

        if not isinstance(image_to_z, list):
            if not isinstance(image_to_z, dict):
                raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                                "as values. Received type {}".format(type(image_to_z)))
            else:
                image_to_z = [image_to_z]

        if not isinstance(collections, list):
            collections = [collections]

        for collection, img_to_z in zip(collections, image_to_z):
            for image in collection.images.values():
                if image.filename not in exclude:
                    if image.filename not in img_to_z.keys():
                        raise ValueError("No z coordinate specified for image {}")
                    else:
                        image.set_z(img_to_z[image.filename])

        self._create_stacks(*collections, exclude=exclude, dilate=dilate)
        # Attribute that holds a Pytorch model
        self._cnn = None
        self._cnn_data_params = None
        self._train_summary = None

    def __len__(self):
        return len(self.calibration_stacks)

    def __repr__(self):
        class_ = 'GdpytCalibrationSet'
        repr_dict = {
                     'Calibration stacks for particle IDs': list(self.calibration_stacks.keys())}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _create_stacks(self, *collections, exclude=[], dilate=True):
        stacks = {}
        ids_in_collects = []
        for i, collection in enumerate(collections):
            if dilate:
                dilation = sqrt(collection.shape_tol + 1)
            else:
                dilation = None
            if i != 0:
                new_id_map = {}
                new_id = max(ids_in_collects) + 1
            for image in collection.images.values():
                if image.filename not in exclude:
                    for particle in image.particles:
                        # For subsequent collections, reassign a unique ID to each particle so stacks from different images don't mix
                        if i != 0:
                            if particle.id not in new_id_map.keys():
                                new_id_map.update({particle.id: new_id})
                                particle.reset_id(new_id)
                                new_id += 1
                            else:
                                particle.reset_id(new_id_map[particle.id])

                        if particle.id not in stacks.keys():
                            new_stack = GdpytCalibrationStack(particle.id, particle.location, dilation=dilation)
                            # For the calibration stack that is used for the conventional method use the filtered template
                            particle.use_raw(False)
                            new_stack.add_particle(particle)
                            stacks.update({particle.id: new_stack})
                        else:
                            stacks[particle.id].add_particle(particle)
                        ids_in_collects.append(particle.id)
            for stack in stacks.values():
                stack.build_layers()

        self._calibration_stacks = stacks

    def infer_z(self, image, function='ccorr', transforms_=(Resize(180), ToTensor())):
        transforms_ = list(transforms_)
        if function not in ['nn', 'cnn']:
            logger.info("Infering image {}".format(image.filename))
            for particle in image.particles:
                stack = self.calibration_stacks[particle.id]
                # Filtered templates are used for correlation calculations
                particle.use_raw(False)
                stack.infer_z(particle, function=function)
        else:
            with torch.no_grad():
                if self.train_summary is None:
                    raise RuntimeError("Calibration set does not have a trained neural net. Use train_cnn "
                                       "before infering using a deep learning model")
                if self._cnn_data_params['normalize']:
                    transforms_.append(Normalize(**self._cnn_data_params['stats']))
                predict_dset = GdpytInceptionDataset(transforms=Compose(transforms_))
                predict_dset.from_image_collections(image, max_size=self._cnn_data_params['max_size'],
                                                    skip_na=self._cnn_data_params['skip_na'])

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    logger.info("Using CUDA for training (Device {})".format(torch.cuda.get_device_name(device)))
                else:
                    logger.info("Using CPU for training")
                    device = torch.device('cpu')

                pred = predict_dset.infer(self.cnn, None,  device=device)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predict_dset.set_sample_z(None, pred)

    def train_cnn(self, epochs, cost_func, normalize_inputs=True, transforms=[Resize(180), ToTensor()],
                  max_sample_size=50, skip_na=True, min_stack_len=10,
                  lr=1e-5, lambda_=1e-3, reg_type=None, batch_size=64, shuffle_batches=True):
        assert isinstance(epochs, int) and epochs > 0

        if reg_type is not None:
            if reg_type.lower() not in ['l2', 'l1']:
                raise ValueError("Regularization can only be L2, L1 or None")

        dataset = GdpytInceptionDataset(transforms=Compose(transforms), normalize=normalize_inputs)
        dataset.from_calib_set(self, max_size=max_sample_size, min_stack_len=min_stack_len, skip_na=skip_na)
        #dataset = GdpytTensorDataset(transforms_=transforms, normalize=normalize_inputs)
        #dataset.from_calib_set(self, max_size=max_sample_size, skip_na=skip_na, min_stack_len=min_stack_len)

        # Save parameters of train data so that the same processing is applied on test data
        self._cnn_data_params = {'normalize': normalize_inputs, 'max_size': max_sample_size, 'skip_na': skip_na,
                                 'shape': dataset.shape, 'stats': dataset.stats}

        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA for training (Device {})".format(torch.cuda.get_device_name(device)))
        else:
            logger.info("Using CPU for training")
            device = torch.device('cpu')

        # Create the Pytoch model
        self._cnn = GdpytInceptionRegressionNet(1000)

        # Set up training input data and parameters
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batches)

        model = self.cnn
        if reg_type is not None:
            if reg_type.lower == 'l2':
                weight_decay = lambda_
            else:
                weight_decay = 0
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if cost_func is not None:
            criterion = eval(cost_func)
        else:
            criterion = nn.L1Loss()
        avg_epoch_loss, std_epoch_loss, model = train_net(model, device, optimizer, criterion, dataloader, epochs=epochs)
        self._train_summary = pd.DataFrame({'Epoch': [i for i in range(epochs)], 'Avg_loss': avg_epoch_loss,
                                            'Sigma_loss': std_epoch_loss})

    def zero_stacks(self, exclude_ids=None):
        for id_, stack in self.calibration_stacks.items():
            if exclude_ids is not None:
                if id_ in exclude_ids:
                    continue
            else:
                stack.set_zero()

    @property
    def calibration_stacks(self):
        return self._calibration_stacks

    @property
    def cnn(self):
        return self._cnn

    @property
    def train_summary(self):
        return self._train_summary