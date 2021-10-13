# import modules
from .GdpytCalibrationStack import GdpytCalibrationStack
from gdpyt.inception_net import GdpytInceptionDataset, GdpytInceptionRegressionNet, train_net
from gdpyt.utils import plotting

import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from matplotlib import pyplot as plt

import logging

logger = logging.getLogger(__name__)

class GdpytCalibrationSet(object):

    def __init__(self, collections, image_to_z, dilate=True, template_padding=0, min_num_layers=None, self_similarity_method='sknccorr',  exclude=[]):
        super(GdpytCalibrationSet, self).__init__()

        self._template_padding = template_padding
        self._particle_ids = None
        self._self_similarity_method = self_similarity_method
        self._min_num_layers = min_num_layers
        self._best_stack_id = None

        # Attribute that holds a Pytorch model
        self._cnn = None
        self.cnn_data_params = None
        self._train_summary = None

        # Ensure image_to_z mapper is the correct data type
        """
        image_to_z maps the calibration image filename to a value in the calibration stack. 
        Notes:
            * the first calibration image should be "..._1.tif" (as opposed to ...0.tif).
            * the calibration stack spans the distance from 0 to Z (where Z is the measurement depth = number of 
            calibration images * z-step per image) by centering the first image at 1 / (2 * number of calibration images) 
            and the last image at 1 - 1 / (2 * number of calibration images)
        """
        if not isinstance(image_to_z, list):
            if not isinstance(image_to_z, dict):
                raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                                "as values. Received type {}".format(type(image_to_z)))
            else:
                image_to_z = [image_to_z]

        if not isinstance(collections, list):
            collections = [collections]

        # Set the z height of each image using image_to_z mapper
        """
        image.set_z: sets the z-height for that image.
        image.add_particles_in_image: for each particle, append the image ID's in which it is identified.
        """
        for collection, img_to_z in zip(collections, image_to_z):
            for image in collection.images.values():
                if image.filename not in exclude:
                    if image.filename not in img_to_z.keys():
                        raise ValueError("No z coordinate specified for image {}")
                    else:

                        # set both the true_z and z value for each image and particle if the particle z-coord is None.
                        image.set_z(img_to_z[image.filename])

                        # sets the "in_images" attribute for GdpytParticle
                        # image.add_particles_in_image() # TODO: Method not working but also not important

        # Create the calibration stacks
        self._create_stacks(*collections, exclude=exclude, dilate=dilate, template_padding=template_padding,
                            self_similarity_method=self_similarity_method)

        # Calculate statistics for all stacks in the set
        self._all_stacks_stats = self.calculate_stacks_stats()

        # Clean the stacks to remove bad stacks
        self._clean_stacks(min_percent_layers=0.1)

        # determine the beset stack
        self.determine_best_stack()

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

    def _create_stacks(self, *collections, exclude=[], exclude_particle_ids=[], dilate=False, template_padding=0,
                       self_similarity_method='sknccorr'):
        """
        Steps:
            1. Loop through all particle ID's and create a GdpytCalibrationStack for each ID.
            2. Build the stack layers for each stack.
            3. Instantiate the GdpytCalibrationSet's stacks attribute.

        Parameters
        ----------
        collections: GdpytImageCollection -- Note multiple collections can be passed in (but this hasn't been tested).
        exclude: image filename's to exclude.
        exclude_particle_ids: particle ID's to exclude.
        dilate: enlarge the template which generally improves correlation -- Note this has largely been deprecated by
        the 'template_padding' attribute in the GdpytImageCollection class.
        template_padding: the template_padding parameter can be used to provide additional padding for calibration
        images to enable sliding of the test particle image template on across the calibration image template. It is not
        used here because template_padding is better used during the initial call to GdpytImageCollection to identify
        and analyze particle images and locations.
        self_similarity_method: cross-correlation method for calibration stack self-similarity assessment.

        Returns
        -------

        """
        stacks = {}
        ids_in_collects = []
        for i, collection in enumerate(collections):

            # dilate (enlarge) the image template (Note: this function has largely been deprecated by the more recent
            # manual 'template_padding' variable in GdpytImageCollection.
            if dilate:
                if isinstance(dilate, float) or isinstance(dilate, int):
                    dilation = dilate
                else:
                    dilation = 1
            else:
                dilation = None

            # create a new collection (i.e. GdpytImageCollection) ID if multiple image collections are passed in.
            if i != 0:
                new_id_map = {}
                new_id = max(ids_in_collects) + 1

            # loop through all particle ID's and create a calibration stack class for each, then build stack layers.
            for image in collection.images.values():
                if image.filename not in exclude:
                    for particle in image.particles:

                        # if multiple collections, reassign a unique ID to each particle so stacks from different images
                        # between collections don't get mixed up.
                        if i != 0:
                            if particle.id not in new_id_map.keys():
                                new_id_map.update({particle.id: new_id})
                                particle.reset_id(new_id)
                                new_id += 1
                            else:
                                particle.reset_id(new_id_map[particle.id])

                        if particle.id in exclude_particle_ids:
                            continue
                        elif particle.id not in stacks.keys():

                            # instantiate GdpytCalibrationStack class for each particle ID.
                            new_stack = GdpytCalibrationStack(particle.id, particle.location, dilation=dilation,
                                                              template_padding=template_padding,
                                                              self_similarity_method=self_similarity_method)

                            # Note on using the filtered or raw image for calibration stacks:
                            # cross-correlation-based calibration stacks: use filtered image (i.e. stacks_use_raw=False)
                            # neural-network-based calibration stacks: use raw image (i.e. stacks_use_raw=True)
                            particle.set_use_raw(collection.stacks_use_raw)
                            new_stack.add_particle(particle)
                            stacks.update({particle.id: new_stack})
                        else:
                            stacks[particle.id].add_particle(particle)
                        ids_in_collects.append(particle.id)

            # once all the particles ID's have been assigned to a stack, build the calibration stack layers.
            for stack in stacks.values():
                stack.build_layers()

        # define the GdpytCalibrationSet's stacks.
        self._calibration_stacks = stacks

    def _clean_stacks(self, min_percent_layers=0.1):
        """
        Remove calibration stacks with too few images or particle stats that differ significantly from others.

        Steps:
            1. Get dataframe of statistics for all stacks in the set.
            2. Get the unique particle ID's in set.
            3. Filter stacks by a minimum number of layers.
                3.1 if min_num_layers is provided, use this as threshold.
                3.2 else, use 75% of the mean number of layers per stack as the threshold.
            4.
        """

        # get set statistics
        df = self.all_stacks_stats

        # get unique particle ID's in set
        all_stacks_uniques = df.particle_id.unique()

        # filter stacks by a minimum number of layers per stack
        if self._min_num_layers:
            df = df.loc[df['layers'] > self._min_num_layers]
        else:
            df = df.loc[df['layers'] > df.layers.mean() * min_percent_layers]

        # get particle id's from filtered dataframe
        passing_stacks_uniques = df.particle_id.unique()

        # get non-passing particle id's
        exclude_particle_ids = list(set(all_stacks_uniques) - set(passing_stacks_uniques))

        # remove stacks from the set
        for id in exclude_particle_ids:

            # delete stack
            del self.calibration_stacks[id]

            # update the particle id's
            self.update_particle_ids()

            logger.warning("Removed calibration stack {} from set.".format(id))

    def _calculate_stack_self_similarity(self):
        for stack in self._calibration_stacks:
            stack.infer_self_similarity(function=self.self_similarity_method)

    def infer_z(self, infer_collection, infer_sub_image=True):
        """

        Parameters
        ----------
        infer_collection: GdpytImageCollection to infer
        infer_sub_image: use sub-image interpolation to calculate z-height

        Returns
        -------
        GdpytImageInference class which holds methods for computing the cross-correlation
        """
        return GdpytImageInference(infer_collection, self, infer_sub_image=infer_sub_image)

    def zero_stacks(self, z_zero=0, offset=0, exclude_ids=None):
        """
        Modify the zero-location (i.e. z = 0) for all stacks in the calibration set.

        The set_zero method adjust the z-coordinate of particles by:
            1. applying a manual offset value.
            2. calculating the z-height where the particle image area is minimized for that stack and using this as the
            offset value.

        Notes:
            * The zero-location does not need to be the focal plane of the imaged particles. It can be arbitrary and/or
            purposeful (e.g. the z-coordinate of a reference feature with respect to the particles' image plane.
            * The zero-location for a calibration set is most usually the focal plane of the particles which
            theoretically is when the particles' image area is minimized. This is not always true in application due to
            pixelation and image thresholding and segmentation algorithms.

        """
        for id_, stack in self.calibration_stacks.items():
            if exclude_ids is not None:
                if id_ in exclude_ids:
                    continue
            else:
                stack.set_zero(z_zero=z_zero, offset=offset)

    def update_particle_ids(self):
        """
        Return a list of the particle ID's in the current calibration set.
        """
        self._particle_ids = list(self.calibration_stacks.keys())

    def determine_best_stack(self):
        """
        Determine the "best" stack to use when the test particle ID is not in the calibration set stack ID's.
        """
        df = self.calculate_stacks_stats().copy()

        # filter by number of layers
        df = df[df['layers'] > df['layers'].max() * 0.98]

        # filter by snr
        df = df[df['avg_snr'] > df['avg_snr'].max() * 0.99]

        # choose the first particle ID if more than one
        particle_ids = df.particle_id.to_numpy(dtype=int, copy=True)
        particle_ids = particle_ids[0]

        self._best_stack_id = particle_ids

    def calculate_stacks_stats(self):
        """
        The calibration set statistics is simply a concatenated dataframe of the per-stack stats. Note that this returns
        the stats for stacks that pass the _clean_stacks filtering. The data for all stacks in the set is stored in the
        _all_stacks_stats attribute/property.

        Steps:
            1. Calculate and get the stats for each stack in the calibration set.
            2. Concatenate each stack to the calibration set dataframe.
        Returns
        -------
        Dataframe of all stacks' stats in the calibration set.
        """
        si = 0
        for stack in self.calibration_stacks.keys():
            calib_stack_data = self.calibration_stacks[stack].calculate_stats()
            calib_stack_data.update({'stack_id': stack, 'p_id': self.calibration_stacks[stack].id})
            if si == 0:
                df_stacks = pd.DataFrame(data=calib_stack_data, index=[stack])
                si += 1
            else:
                new_stacks = pd.DataFrame(data=calib_stack_data, index=[stack])
                df_stacks = pd.concat([df_stacks, new_stacks])
        return df_stacks

    def plot_stacks_self_similarity(self, min_num_layers=0):
        return plotting.plot_stacks_self_similarity(calib_set=self, min_num_layers=min_num_layers)

    def train_cnn(self,
                  epochs,
                  cost_func,
                  normalize_dataset=True,
                  normalize_per_sample=False,
                  transforms=[Resize(180), ToTensor()],
                  aux_logits=None,
                  max_sample_size=50,
                  skip_na=True,
                  min_stack_len=10,
                  lr=1e-5,
                  lambda_=1e-3,
                  reg_type=None,
                  batch_size=64,
                  shuffle_batches=True):
        """

        Parameters
        ----------
        epochs
        cost_func
        normalize_dataset
        normalize_per_sample
        transforms
        aux_logits
        max_sample_size
        skip_na
        min_stack_len
        lr
        lambda_
        reg_type
        batch_size
        shuffle_batches

        Returns
        -------

        """
        assert isinstance(epochs, int) and epochs > 0

        if reg_type is not None:
            if reg_type.lower() not in ['l2', 'l1']:
                raise ValueError("Regularization can only be L2, L1 or None")

        dataset = GdpytInceptionDataset(transforms=Compose(transforms), aux_logits=aux_logits,
                                        normalize_dataset=normalize_dataset,
                                        normalize_per_sample=normalize_per_sample)
        dataset.from_calib_set(self, max_size=max_sample_size, min_stack_len=min_stack_len, skip_na=skip_na)
        # dataset = GdpytTensorDataset(transforms_=transforms, normalize=normalize_inputs)
        # dataset.from_calib_set(self, max_size=max_sample_size, skip_na=skip_na, min_stack_len=min_stack_len)

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
        avg_epoch_loss, std_epoch_loss, model = train_net(model, device, optimizer, criterion, dataloader,
                                                          epochs=epochs)
        self._train_summary = pd.DataFrame({'Epoch': [i for i in range(epochs)], 'Avg_loss': avg_epoch_loss,
                                            'Sigma_loss': std_epoch_loss})

    def plot_train_summary(self):
        fig, axes = plt.subplots(nrows=2)
        ax = axes.ravel()
        ax[0].plot(self.train_summary['Epoch'], self.train_summary['Avg_loss'], label='Avg loss')
        ax[1].plot(self.train_summary['Epoch'], self.train_summary['Sigma_loss'], label=r'$\sigma$ loss')
        plt.tight_layout()
        plt.show()

    @property
    def calibration_stacks(self):
        return self._calibration_stacks

    @property
    def particle_ids(self):
        self.update_particle_ids()
        return self._particle_ids

    @property
    def best_stack_id(self):
        return self._best_stack_id

    @property
    def template_padding(self):
        return self._template_padding

    @property
    def min_num_layers(self):
        return self._min_num_layers

    @property
    def all_stacks_stats(self):
        return self._all_stacks_stats

    @property
    def calib_set_stats(self):
        return self.calculate_stacks_stats()

    @property
    def self_similarity_method(self):
        return self._self_similarity_method

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

    def _cross_correlation_inference(self, function, use_stack=None, min_cm=0, skip_particle_ids=[]):
        logger.warning('cc inference min_cm {}'.format(min_cm))
        if function.lower() not in ['ccorr', 'nccorr', 'znccorr', 'barnkob_ccorr', 'bccorr', 'sknccorr']:
            raise ValueError("{} is not implemented or a valid function".format(function))

        for image in self.collection.images.values():
            logger.info("Infering image {}".format(image.filename))

            max_stack_distance = 15

            particles_s = [particle_s for particle_s in image.particles]

            for particle in image.particles:

                particle_locations_in_image = [list(p.location) for p in particles_s if p.id != particle.id]

                if len(particle_locations_in_image) > 2:
                    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(particle_locations_in_image)
                    distance, index = nneigh.kneighbors(np.array(particle.location).reshape(1, -1))
                    distance = distance[0][0].astype(int)
                    index = index[0][0].astype(int)
                else:
                    distance = 100
                    index = 0

                if use_stack is not None:
                    stack = self.calib_set.calibration_stacks[use_stack]

                elif particle.id in self.calib_set.particle_ids:
                    stack = self.calib_set.calibration_stacks[particle.id]

                elif distance < max_stack_distance and particles_s[index].id in self.calib_set.particle_ids:
                    stack = self.calib_set.calibration_stacks[particles_s[index].id]

                else:
                    # if nothing else, choose the best stack id
                    stack = self.calib_set.calibration_stacks[self.calib_set.best_stack_id]

                # set the stack ID used for z-inference
                particle.set_inference_stack_id(stack.id)

                # infer z
                stack.infer_z(particle, function=function, min_cm=min_cm, infer_sub_image=self._infer_sub_image) # Filtered templates are used for correlation calculations

    def ccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('ccorr', use_stack=use_stack, min_cm=min_cm)

    def nccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('nccorr', use_stack=use_stack, min_cm=min_cm)

    def znccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('znccorr', use_stack=use_stack, min_cm=min_cm)

    def bccorr(self, use_stack=None, min_cm=0):
        self._cross_correlation_inference('barnkob_ccorr', use_stack=use_stack, min_cm=min_cm)

    def sknccorr(self, use_stack=None, min_cm=0, skip_particle_ids=[]):
        self._cross_correlation_inference('sknccorr', use_stack=use_stack, min_cm=min_cm, skip_particle_ids=skip_particle_ids)

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