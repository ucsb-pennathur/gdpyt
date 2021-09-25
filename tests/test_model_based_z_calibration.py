"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytImageCollection, GdpytSetup, GdpytCharacterize
from gdpyt.utils.datasets import dataset_unpacker
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square

# ----- ----- ----- ----- TEST DATASET UNPACKER ----- ----- ----- ----- ----- ----- -----
test_dataset = '20X_1Xmag_5.61umPink_HighInt'

if test_dataset == 'JP-EXF01-20':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=0, number_of_images=50).unpack()
elif test_dataset == '20X_1Xmag_5.61umPink_HighInt':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()
elif test_dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration').unpack()

calib_col = GdpytCharacterize.test(calib_settings, None, return_variables='calibration_collection')

# plot the particle image PSF-model-based z-calibration function
calib_col.plot_gaussian_ax_ay(plot_type='all', p_inspect=[0])
plt.suptitle(calib_settings.outputs.save_id_string + '_all_p_ids')
plt.tight_layout()
if calib_settings.outputs.save_plots is True:
    savefigpath = join(calib_settings.outputs.results_path,
                       calib_settings.outputs.save_id_string + '_Gaussian_fit_axy_all_pids.png')
    plt.savefig(fname=savefigpath, bbox_inches='tight')
    #    plt.close()
    # if calib_settings.outputs.show_plots:
    plt.show()

calib_col.plot_gaussian_ax_ay(plot_type='mean', p_inspect=[0])
plt.suptitle(calib_settings.outputs.save_id_string + '_mean_p_ids')
plt.tight_layout()
if calib_settings.outputs.save_plots is True:
    savefigpath = join(calib_settings.outputs.results_path,
                       calib_settings.outputs.save_id_string + '_Gaussian_fit_axy_mean_pids.png')
    plt.savefig(fname=savefigpath, bbox_inches='tight')
    #    plt.close()
    # if calib_settings.outputs.show_plots:
    plt.show()

for img in calib_col.images.values():
    for p in img.particles:
        calib_col.plot_gaussian_ax_ay(plot_type='one', p_inspect=[p.id])
        plt.suptitle(calib_settings.outputs.save_id_string + 'p_id_{}'.format(p.id))
        plt.tight_layout()
        if calib_settings.outputs.save_plots is True:
            savefigpath = join(calib_settings.outputs.results_path, calib_settings.outputs.save_id_string + '_Gaussian_fit_axy_pid{}.png'.format(p.id))
            plt.savefig(fname=savefigpath, bbox_inches='tight')
            #    plt.close()
            #if calib_settings.outputs.show_plots:
            plt.show()

j=1