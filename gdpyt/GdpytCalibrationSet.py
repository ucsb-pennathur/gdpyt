from matplotlib import pyplot as plt

from .GdpytCalibrationStack import GdpytCalibrationStack
from gdpyt.inception_net import GdpytInceptionDataset, GdpytInceptionRegressionNet, train_net
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
                        image.add_particles_in_image()

        self._create_stacks(*collections, exclude=exclude, dilate=dilate)
        # Attribute that holds a Pytorch model
        self._cnn = None
        self.cnn_data_params = None
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
                if isinstance(dilate, float) or isinstance(dilate, int):
                    dilation = dilate
                else:
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
                            # cross-correlation-based calibration stacks: use filtered image (i.e. stacks_use_raw=False)
                            # neural-network-based calibration stacks: use raw image (i.e. stacks_use_raw=True)
                            particle.use_raw(collection.stacks_use_raw)
                            new_stack.add_particle(particle)
                            stacks.update({particle.id: new_stack})
                        else:
                            stacks[particle.id].add_particle(particle)
                        ids_in_collects.append(particle.id)
            for stack in stacks.values():
                stack.build_layers()

        self._calibration_stacks = stacks

    # def fill_missing_levels(self, exclude_stacks=None):
    #     logger.info("Filling missing calibration levels..")
    #     for stack_id, stack in self.calibration_stacks.items():
    #         if exclude_stacks is not None:
    #             if stack_id in exclude_stacks:
    #                 continue
    #         else:
    #             lvls_in_stack = list(stack.get_layers().keys())
    #             missing_levels = OrderedDict()
    #             for z, img_name in sorted(self.z_levels).items():
    #                 if z not in lvls_in_stack:
    #                     missing_levels.update({z: img_name})
    #
    #             stack.fill_levels(missing_levels)

    def infer_z(self, infer_collection, infer_sub_image=True):
        return GdpytImageInference(infer_collection, self, infer_sub_image=infer_sub_image)

    def train_cnn(self, epochs, cost_func, normalize_dataset=True, normalize_per_sample=False,
                  transforms=[Resize(180), ToTensor()], aux_logits=None,
                  max_sample_size=50, skip_na=True, min_stack_len=10,
                  lr=1e-5, lambda_=1e-3, reg_type=None, batch_size=64, shuffle_batches=True):
        assert isinstance(epochs, int) and epochs > 0

        if reg_type is not None:
            if reg_type.lower() not in ['l2', 'l1']:
                raise ValueError("Regularization can only be L2, L1 or None")

        dataset = GdpytInceptionDataset(transforms=Compose(transforms), aux_logits=aux_logits,
                                        normalize_dataset=normalize_dataset,
                                        normalize_per_sample=normalize_per_sample)
        dataset.from_calib_set(self, max_size=max_sample_size, min_stack_len=min_stack_len, skip_na=skip_na)
        #dataset = GdpytTensorDataset(transforms_=transforms, normalize=normalize_inputs)
        #dataset.from_calib_set(self, max_size=max_sample_size, skip_na=skip_na, min_stack_len=min_stack_len)

        # Save parameters of train data so that the same processing is applied on test data
        self.cnn_data_params = {'normalize_dataset': normalize_dataset, 'normalize_per_sample': normalize_per_sample,
                                 'max_size': max_sample_size, 'skip_na': skip_na,
                                 'shape': dataset.shape, 'stats': dataset.stats}

        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA for training (Device {})".format(torch.cuda.get_device_name(device)))
        else:
            logger.info("Using CPU for training")
            device = torch.device('cpu')

        # Create the Pytorch model
        if aux_logits is None:
            # Without auxiliary loss always make 1000 classes
            self._cnn = GdpytInceptionRegressionNet(1000)
        else:
            n_classes_logits = len(aux_logits) + 1
            self._cnn = GdpytInceptionRegressionNet(n_classes_logits, aux_logits=True)

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
            criterion = cost_func
        else:
            criterion = nn.L1Loss()
        avg_epoch_loss, std_epoch_loss, model = train_net(model, device, optimizer, criterion, dataloader, epochs=epochs)
        self._train_summary = pd.DataFrame({'Epoch': [i for i in range(epochs)], 'Avg_loss': avg_epoch_loss,
                                            'Sigma_loss': std_epoch_loss})

    def plot_train_summary(self):
        fig, axes = plt.subplots(nrows=2)
        ax = axes.ravel()
        ax[0].plot(self.train_summary['Epoch'], self.train_summary['Avg_loss'], label='Avg loss')
        ax[1].plot(self.train_summary['Epoch'], self.train_summary['Sigma_loss'], label=r'$\sigma$ loss')
        plt.tight_layout()
        plt.show()

    def zero_stacks(self, offset=0, exclude_ids=None):
        for id_, stack in self.calibration_stacks.items():
            if exclude_ids is not None:
                if id_ in exclude_ids:
                    continue
            else:
                stack.set_zero(offset=offset)

    @property
    def calibration_stacks(self):
        return self._calibration_stacks

    @property
    def particle_ids(self):
        return list(self.calibration_stacks.keys())

    @property
    def cnn(self):
        return self._cnn

    @property
    def train_summary(self):
        return self._train_summary


class GdpytImageInference(object):

    def __init__(self, infer_collection, calib_set, infer_sub_image):
        self.collection = infer_collection
        assert isinstance(calib_set, GdpytCalibrationSet)
        self.calib_set = calib_set
        self._infer_sub_image = infer_sub_image

    def _cross_correlation_inference(self, function, use_stack=None, min_cm=0):
        logger.warning('cc inference min_cm {}'.format(min_cm))
        if function.lower() not in ['ccorr', 'nccorr', 'znccorr', 'barnkob_ccorr']:
            raise ValueError("{} is not implemented or a valid function".format(function))

        for image in self.collection.images.values():
            logger.info("Infering image {}".format(image.filename))
            for particle in image.particles:
                if use_stack is None:
                    if particle.id < len(self.calib_set.calibration_stacks):
                        stack = self.calib_set.calibration_stacks[particle.id]
                    else:
                        stack = self.calib_set.calibration_stacks[0]
                else:
                    stack = self.calib_set.calibration_stacks[use_stack]

                stack.infer_z(particle, function=function, min_cm=min_cm, infer_sub_image=self._infer_sub_image) # Filtered templates are used for correlation calculations

    def ccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('ccorr', use_stack=use_stack, min_cm=min_cm)

    def nccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('nccorr', use_stack=use_stack, min_cm=min_cm)

    def znccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('znccorr', use_stack=use_stack, min_cm=min_cm)

    def bccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('barnkob_ccorr', use_stack=use_stack, min_cm=min_cm)

    def cnn(self, transforms_=(Resize(180), ToTensor()), pretrained=None):
        with torch.no_grad():
            if self.calib_set.train_summary is None:
                raise RuntimeError("Calibration set does not have a trained neural net. Use train_cnn "
                                   "before infering using a deep learning model")
            if self.calib_set.cnn_data_params['normalize_dataset']:
                logger.info(
                    "Setting normalization parameters of prediction set to {}".format(self.calib_set.cnn_data_params['stats']))
                transforms_.append(Normalize(**self.calib_set.cnn_data_params['stats']))
            predict_dset = GdpytInceptionDataset(transforms=Compose(transforms_),
                                                 normalize_per_sample=self.calib_set.cnn_data_params['normalize_per_sample'],
                                                 normalize_dataset=self.calib_set.cnn_data_params['normalize_dataset'])
            predict_dset.from_image_collections(self.collection, template_shape=self.calib_set.cnn_data_params['shape'],
                                                max_size=self.calib_set.cnn_data_params['max_size'],
                                                skip_na=self.calib_set.cnn_data_params['skip_na'])

            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Using CUDA for inference (Device {})".format(torch.cuda.get_device_name(device)))
            else:
                logger.info("Using CPU for training")
                device = torch.device('cpu')

            if pretrained is not None:
                logger.warning("Using pretrained model to infer...")
                assert isinstance(pretrained, torch.nn.Module)
                predictor_cnn = pretrained
            else:
                predictor_cnn = self.calib_set.cnn

            pred = predict_dset.infer(predictor_cnn, None, device=device)
            if isinstance(pred, tuple):
                pred = pred[0]
            predict_dset.set_sample_z(None, pred)

    @property
    def infer_sub_image(self):
        return self._infer_sub_image