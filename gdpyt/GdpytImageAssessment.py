

# imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ma

plt.style.use(['science', 'ieee', 'std-colors'])
from skimage.filters import threshold_otsu

from .GdpytImage import GdpytImage

from os.path import join, isdir
from os import listdir
from collections import OrderedDict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class GdpytImageAssessment(object):

    def __init__(self, folder, filetype, file_basestring):

        super(GdpytImageAssessment, self).__init__()

        if not isdir(folder):
            raise ValueError("Specified folder {} does not exist".format(folder))

        # properties of the image collection
        self._folder = folder
        self._filetype = filetype
        self._file_basestring = file_basestring

        # add images
        self._find_files()
        self._add_images()

        # assess images
        self._assess_images()
        self._plot_image_assessment()

    def _find_files(self):
        """
        Identifies all files of filetype filetype in folder
        :return:
        """
        all_files = listdir(self._folder)
        number_of_files = 0
        save_files = []
        for file in all_files:
            if file.endswith(self._filetype):
                save_files.append(file)
                number_of_files += 1

        save_files = sorted(save_files,
                            key=lambda filename: float(filename.split(self._file_basestring)[-1].split('.')[0]))

        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(save_files), self._filetype, self._folder))
        # Save all the files of the right filetype in this attribute
        self._files = save_files

    def _add_images(self):
        images = OrderedDict()
        frame = 0
        for file in self._files:
            img = GdpytImage(join(self._folder, file), frame=frame)
            images.update({img.filename: img})
            logger.warning('Loaded image {}'.format(img.filename))
            frame += 1
        self._images = images

    def _assess_images(self):
        frames = []
        means = []
        stds = []
        mins = []
        maxs = []
        otsu_threshs = []
        signal_means = []
        noise_means = []
        noise_stds = []

        for name, gdpyt_image in self._images.items():
            img = gdpyt_image.raw

            frames.append(gdpyt_image.frame)
            means.append(np.mean(img))
            stds.append(np.std(img))
            mins.append(np.min(img))
            maxs.append(np.max(img))

            # threshold analysis
            thresh = threshold_otsu(img)
            mask_signal = img < thresh
            img_signal = ma.array(img, mask=mask_signal)
            mask_noise = img > thresh
            img_noise = ma.array(img, mask=mask_noise)


            otsu_threshs.append(thresh)
            signal_means.append(img_signal.mean())
            noise_means.append(img_noise.mean())
            noise_stds.append(img_noise.std())

        data = np.array([frames, means, stds, mins, maxs, otsu_threshs, signal_means, noise_means, noise_stds]).T

        df = pd.DataFrame(data, columns=['frames', 'means', 'stds', 'mins', 'maxs',
                                         'otsu_threshs', 'signal_means', 'bkg_means', 'bkg_stds'])

        df['noise_floor'] = df['bkg_means'] + df['bkg_stds'] * 2

        self._image_stats = df

    def _plot_image_assessment(self):
        df = self._image_stats

        fig, ax2 = plt.subplots()

        ax2.plot(df.frames, df.otsu_threshs, label='otsu')
        ax2.plot(df.frames, df.signal_means, label='signal')
        ax2.plot(df.frames, df.bkg_means, label='bkg')
        ax2.plot(df.frames, df.noise_floor, label='noise ceiling')

        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Frame')
        ax2.set_title('Noise ceiling = {} +/- {}'.format(np.round(df.bkg_means.min(), 1),
                                                         np.round(df.bkg_stds.min() * 2, 1)))
        ax2.grid(alpha=0.125)
        ax2.legend()

        plt.tight_layout()
        plt.show()