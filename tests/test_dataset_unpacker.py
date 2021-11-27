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
"""
sweep parameters of interest:
median filter size: [0, 3, 5, 7]
particle density: ['1e-3', '2.5e-3', '5e-3', '7.5e-3', '10e-3']
test template padding: [0, 2, 4, 6] w/ calib template padding of 6
baseline images: 
    calib_baseline = ['calib_-35.0.tif', 'calib_-25.0.tif', 'calib_-15.0.tif', 'calib_-5.0.tif', 'calib_5.0.tif']
    test_particle_id = ['calib_-34.77387.tif', 'calib_-25.12563.tif', 'calib_-15.07538.tif', 'calib_-5.02513.tif', 'calib_5.02513.tif']
"""
test_dataset = 'synthetic_overlap_noise-level'
particle_distribution = 'random'
num_test_images = 200
nl = 1
sweep_method = 'particle density'
sweep_params = ['1e-3', '2.5e-3', '5e-3', '7.5e-3', '10e-3']

# experiments
known_z = None

# generate calibration collection and calibration set
"""
calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                          static_templates=False, single_particle_calibration=True, hard_baseline=True,
                                          particles_overlapping=True, particle_distribution=particle_distribution).unpack()
calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')
"""

for sweep_param in sweep_params:

    if test_dataset == 'synthetic_experiment':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution, particle_density=sweep_param,
                                          single_particle_calibration=False, static_templates=True,
                                          hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                          sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=None,
                                         number_of_images=num_test_images,
                                         particle_distribution=particle_distribution, particle_density=sweep_param,
                                         single_particle_calibration=False, static_templates=True,
                                         hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                         sweep_param=sweep_param).unpack()
    elif test_dataset == 'JP-EXF01-20':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                          number_of_images=50, particle_distribution='Dataset_I',
                                          particle_density='361', static_templates=False, hard_baseline=False,
                                          single_particle_calibration=True, particles_overlapping=False).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=2,
                                         number_of_images=num_test_images, particle_distribution='Dataset_I',
                                         particle_density='361', static_templates=False, hard_baseline=False,
                                         single_particle_calibration=True, particles_overlapping=False).unpack()
    elif test_dataset == 'synthetic_overlap_noise-level':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution, particle_density=sweep_param,
                                          single_particle_calibration=True, static_templates=False,
                                          hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                          sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                         number_of_images=num_test_images,
                                         particle_distribution=particle_distribution, particle_density=sweep_param,
                                         single_particle_calibration=True, static_templates=False,
                                         hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                         sweep_param=sweep_param).unpack()
    elif test_dataset == '11.06.21_z-micrometer-v2':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, particle_distribution=particle_distribution,
                                          sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    elif test_dataset == '11.09.21_z-micrometer-v3':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, particle_distribution=particle_distribution,
                                          sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param, known_z=known_z).unpack()
    elif test_dataset == '20X_1Xmag_5.61umPink_HighInt':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True).unpack()
    elif test_dataset == '11.02.21-BPE_Pressure_Deflection_20X':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    elif test_dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    elif test_dataset == '10X_1Xmag_5.1umNR_HighInt_0.06XHg':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True).unpack()
    elif test_dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=False, single_particle_calibration=False, hard_baseline=True).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=False, single_particle_calibration=False, hard_baseline=True).unpack()
    elif test_dataset == '20X_1Xmag_0.87umNR_on_Silpuran':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=False, single_particle_calibration=False,
                                          hard_baseline=True).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=False, single_particle_calibration=False,
                                         hard_baseline=True).unpack()
    elif test_dataset == '20X_1Xmag_0.87umNR':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False,
                                          hard_baseline=True).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False,
                                         hard_baseline=True).unpack()
    else:
        raise ValueError("No dataset found.")

    # ----- ----- ----- ----- TEST ----- ----- ----- ----- ----- ----- -----
    GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None, return_variables=None)


# ----- ----- ----- ----- SPC Random Density --- -- ----- ----- ----- ----- ----- -----

# generate calibration collection and calibration set
sweep_params = ['7.5e-3', '10e-3']
for sweep_param in sweep_params:
    calib_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=sweep_param,
                                      single_particle_calibration=False, static_templates=True,
                                      hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                      sweep_param=sweep_param).unpack()
    test_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                     number_of_images=num_test_images,
                                     particle_distribution=particle_distribution, particle_density=sweep_param,
                                     single_particle_calibration=False, static_templates=True,
                                     hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                     sweep_param=sweep_param).unpack()
    GdpytCharacterize.test(calib_settings2, test_settings2, return_variables=None)

"""# ----- ----- ----- ----- Static Grid Overlap Random Z --- -- ----- ----- ----- ----- ----- -----
particle_distribution = 'grid-random-z'

# generate calibration collection and calibration set
for sweep_param in sweep_params:
    calib_settings3 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                      static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                      particles_overlapping=True, particle_distribution=particle_distribution).unpack()
    test_settings3 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                     number_of_images=num_test_images,
                                     particle_distribution=particle_distribution, particle_density=sweep_param,
                                     single_particle_calibration=False, static_templates=True,
                                     hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                     sweep_param=sweep_param).unpack()
    GdpytCharacterize.test(calib_settings3, test_settings3, return_variables=None)

# ----- ----- ----- ----- SPC Grid Overlap Random Z --- -- ----- ----- ----- ----- ----- -----
particle_distribution = 'grid-random-z'

# generate calibration collection and calibration set
for sweep_param in sweep_params:
    calib_settings4 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                      static_templates=False, single_particle_calibration=True, hard_baseline=True,
                                      particles_overlapping=True, particle_distribution=particle_distribution).unpack()
    test_settings4 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                     number_of_images=num_test_images,
                                     particle_distribution=particle_distribution, particle_density=sweep_param,
                                     single_particle_calibration=True, static_templates=False,
                                     hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                     sweep_param=sweep_param).unpack()
    GdpytCharacterize.test(calib_settings4, test_settings4, return_variables=None)
"""

# ----- ----- ----- ----- SWEEP Density --- -- ----- ----- ----- ----- ----- -----
"""for sweep_param in sweep_params:

    calib_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=1,
                                       number_of_images=num_test_images,
                                       particle_distribution=particle_distribution, particle_density=sweep_param,
                                       single_particle_calibration=True, static_templates=False,
                                       hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                       sweep_param=sweep_param).unpack()
    test_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=1,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=sweep_param,
                                      single_particle_calibration=True, static_templates=False,
                                      hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                      sweep_param=sweep_param).unpack()

    GdpytCharacterize.test(calib_settings2, test_settings2, return_variables=None)"""

"""
# ----- ----- ----- ----- SWEEP BASELINE - SPC ----- ----- ----- ----- ----- ----- -----
sweep_method = 'baseline_image'
calib_baseline = ['calib_-35.0.tif', 'calib_-25.0.tif', 'calib_-15.0.tif', 'calib_-5.0.tif', 'calib_5.0.tif']
sweep_params = calib_baseline

for sweep_param in sweep_params:

    calib_settings3 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                       number_of_images=num_test_images,
                                       particle_distribution=particle_distribution, particle_density=None,
                                       single_particle_calibration=True, static_templates=False,
                                       hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                       sweep_param=sweep_param).unpack()
    test_settings3 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=2,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=None,
                                      single_particle_calibration=True, static_templates=False,
                                      hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                      sweep_param=sweep_param).unpack()

    GdpytCharacterize.test(calib_settings3, test_settings3, return_variables=None)


# ----- ----- ----- ----- SWEEP SAME ID THRESHOLD - Static ----- ----- ----- ----- ----- ----- -----
sweep_method = 'same_id_thresh'
sweep_params = [5, 10, 15, 20]

for sweep_param in sweep_params:
    calib_settings4 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=None,
                                      single_particle_calibration=False, static_templates=True,
                                      hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                      sweep_param=sweep_param).unpack()
    test_settings4 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=2,
                                     number_of_images=num_test_images,
                                     particle_distribution=particle_distribution, particle_density=None,
                                     single_particle_calibration=False, static_templates=True,
                                     hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                     sweep_param=sweep_param).unpack()

    GdpytCharacterize.test(calib_settings4, test_settings4, return_variables=None)

# ----- ----- ----- ----- SWEEP SAME ID THRESHOLD - SPC ----- ----- ----- ----- ----- ----- -----
sweep_method = 'same_id_thresh'
sweep_params = [5, 10, 15, 20]

for sweep_param in sweep_params:

    calib_settings5 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                       number_of_images=num_test_images,
                                       particle_distribution=particle_distribution, particle_density=None,
                                       single_particle_calibration=True, static_templates=False,
                                       hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                       sweep_param=sweep_param).unpack()
    test_settings5 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=2,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=None,
                                      single_particle_calibration=True, static_templates=False,
                                      hard_baseline=False, particles_overlapping=True, sweep_method=sweep_method,
                                      sweep_param=sweep_param).unpack()

    GdpytCharacterize.test(calib_settings5, test_settings5, return_variables=None)"""

print('Analysis completed without errors.')