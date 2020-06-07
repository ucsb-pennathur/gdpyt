from .GdpytCalibrationStack import GdpytCalibrationStack
from gdpyt.similarity.nn import GdpytNet, train_net
from gdpyt.utils.nn import GdpytTensorDataset
import pandas as pd
import torch
import torch.nn as nn
from math import sqrt
import logging

logger = logging.getLogger(__name__)

class GdpytCalibrationSet(object):

    def __init__(self, collection, image_to_z, exclude=[], dilate=True):
        super(GdpytCalibrationSet, self).__init__()
        self._collection = collection

        if not isinstance(image_to_z, dict):
            raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                            "as values. Received type {}".format(type(image_to_z)))

        for image in collection.images.values():
            if image.filename not in exclude:
                if image.filename not in image_to_z.keys():
                    raise ValueError("No z coordinate specified for image {}")
                else:
                    image.set_z(image_to_z[image.filename])
        self._create_stacks(exclude=exclude, dilate=dilate)
        # Attribute that holds a Pytorch model
        self._cnn = None
        self._cnn_data_params = None
        self._train_summary = None

    def __len__(self):
        return len(self.calibration_stacks)

    def __repr__(self):
        class_ = 'GdpytCalibrationSet'
        repr_dict = {'GdpytImageCollection': self._collection.folder,
                     'Calibration stacks for particle IDs': list(self.calibration_stacks.keys())}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _create_cnn_model(self, inp_shape):
        if not len(inp_shape) == 3:
            raise TypeError("CNN input must be 3D (C X H X W)")
        self._cnn = GdpytNet(inp_shape, **self._cnn_params)

    def _create_stacks(self, exclude=[], dilate=True):
        stacks = {}
        if dilate:
            dilation = sqrt(self.collection.shape_tol + 1)
        else:
            dilation = None
        for image in self.collection.images.values():
            if image.filename not in exclude:
                for particle in image.particles:
                    if particle.id not in stacks.keys():
                        new_stack = GdpytCalibrationStack(particle.id, particle.location, dilation=dilation)
                        # For the calibration stack that is used for the conventional method use the filtered template
                        particle.use_raw(False)
                        new_stack.add_particle(particle)
                        stacks.update({particle.id: new_stack})
                    else:
                        stacks[particle.id].add_particle(particle)

        for stack in stacks.values():
            stack.build_layers()

        self._calibration_stacks = stacks

    def create_cnn(self, n_conv_layers=4, n_linear_layers=2, kernel_size=5, n_filters_init=16, max_pool_params=None,
                   batch_norm=(0, 1, 2)):
        if self._cnn is not None:
            logger.warning("The existing CNN model is being overwritten once training is started")
        self._cnn_params = dict(n_conv_layers=n_conv_layers, n_linear_layers=n_linear_layers,
                             kernel_size=kernel_size, n_filters_init=n_filters_init,
                             max_pool_params=max_pool_params, batch_norm=batch_norm)

    def infer_z(self, image, function='ccorr'):
        if function not in ['nn', 'cnn']:
            for particle in image.particles:
                stack = self.calibration_stacks[particle.id]
                stack.infer_z(particle, function=function)
        else:
            if self.train_summary is None:
                raise RuntimeError("Calibration set does not have a trained neural net. Use create_cnn and train_cnn "
                                   "before infering using a deep learning model")
            predict_dset = GdpytTensorDataset(normalize=self._cnn_data_params['normalize'])
            predict_dset.from_image_collection(image, ref_shape=self._cnn_data_params['shape'],
                                               max_size=self._cnn_data_params['max_size'],
                                               skip_na=self._cnn_data_params['skip_na'])

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            pred = predict_dset.infer(self.cnn, None,  device=device)
            predict_dset.set_sample_z(None, pred)

    def train_cnn(self, epochs, normalize_inputs=True, transforms=None, max_sample_size=50, skip_na=True, min_stack_len=10,
                  lr=1e-5, lambda_=1e-3, reg_type=None, batch_size=64, shuffle_batches=True):
        assert isinstance(epochs, int) and epochs > 0

        if reg_type is not None:
            if reg_type.lower() not in ['l2', 'l1']:
                raise ValueError("Regularization can only be L2, L1 or None")

        dataset = GdpytTensorDataset(transforms_=transforms, normalize=normalize_inputs)
        dataset.from_calib_set(self, max_size=max_sample_size, skip_na=skip_na, min_stack_len=min_stack_len)

        # Save parameters of train data so that the same processing is applied on test data
        self._cnn_data_params = {'normalize': normalize_inputs, 'max_size': max_sample_size, 'skip_na': skip_na,
                                 'shape': dataset.shape}

        # Create the Pytoch model
        self._create_cnn_model((1,) + dataset.shape)

        # Set up training input data and parameters
        dataloader = dataset.return_dataloader(batch_size=batch_size, shuffle=shuffle_batches)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA for training. Device: {}".format(torch.cuda.get_device_name(device)))
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")

        model = self.cnn
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()
        avg_epoch_loss, std_epoch_loss, model = train_net(model, device, optimizer, criterion, dataloader,
                                                            epochs=epochs, lambda_=lambda_, reg_type=reg_type)
        self._train_summary = pd.DataFrame({'Epoch': [i for i in range(epochs)], 'Avg_loss': avg_epoch_loss,
                                            'Sigma_loss': std_epoch_loss})

    @property
    def collection(self):
        return self._collection

    @property
    def calibration_stacks(self):
        return self._calibration_stacks

    @property
    def cnn(self):
        return self._cnn

    @property
    def train_summary(self):
        return self._train_summary