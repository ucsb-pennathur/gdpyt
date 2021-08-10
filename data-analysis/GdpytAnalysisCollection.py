import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class GdpytAnalysisCollection(object):

    def __init__(self, folder, filetype, crop_specs=None, processing_specs=None, thresholding_specs=None,
                 background_subtraction=None, min_particle_size=None, max_particle_size=None,
                 shape_tol=0.2, overlap_threshold=0.3, exclude=[]):
        super(GdpytAnalysisCollection, self).__init__()
        if not isdir(folder):
            raise ValueError("Specified folder {} does not exist".format(folder))

        self._folder = folder
        self._filetype = filetype
        self._find_files(exclude=exclude)
        self._add_images()

        # Define the cropping done on all the images in this collection
        if crop_specs is None:
            self._crop_specs = {}
        else:
            self._crop_specs = crop_specs
            self.crop_images()

        # Define the background image subtraction method
        if background_subtraction is None:
            self._background_subtraction = {}
        else:
            self._background_subtraction = background_subtraction
            self._background_subtract()

        # Define the processing done on all the images in this collection
        if processing_specs is None:
            self._processing_specs = {}
        else:
            self._processing_specs = processing_specs

        # Define the thresholding done on all the images in this collection
        if thresholding_specs is None:
            self._thresholding_specs = {'otsu': []}
        else:
            self._thresholding_specs = thresholding_specs

        # Minimum and maximum particle size for image in this collection
        self._min_particle_size = min_particle_size
        self._max_particle_size = max_particle_size

        # Shape tolerance
        if shape_tol is not None:
            if not 0 < shape_tol < 1:
                raise ValueError("Shape tolerance parameter shape_tol must be between 0 and 1")
        self._shape_tol = shape_tol

        # Overlap threshold for merging particles
        self._overlap_threshold = overlap_threshold

        self.filter_images()
        self.identify_particles()
        # self.uniformize_particle_ids()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        key = list(self.images.keys())[item]
        return self.images[key]

    def __repr__(self):
        class_ = 'GdpytImageCollection'
        repr_dict = {'Folder': self.folder,
                     'Filetype': self.filetype,
                     'Number of images': len(self),
                     'Min. and max. particle sizes': [self._min_particle_size, self._max_particle_size],
                     'Shape tolerance': self._shape_tol,
                     'Preprocessing': self._processing_specs}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str