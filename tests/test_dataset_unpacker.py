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

test_dataset = '02.07.22_membrane_characterization'  #
particle_distribution = 'meta'
num_test_images = 50
nl = None

sweep_method = 'testset'  # 'subset_mean'

# experiments
known_z = None

# generate calibration collection and calibration set
"""calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                  number_of_images=num_test_images,
                                  particle_distribution=particle_distribution, particle_density=None,
                                  single_particle_calibration=False, static_templates=True,
                                  hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                  sweep_param='gen_cal', use_stack_id=None).unpack()
calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')"""


# GDPyT test collection
sweep_params = [[1, 1], [1, 2], [1, 3], [3, 1], [3, 2], [3, 3], [5, 1], [5, 2], [5, 3], [7, 1], [7, 2], [7, 3], [9, 1]]  # ['0', '0.25', 'second_0.25', '2.25', '4.25', '6.25', '8.25', '10.25', '11.25', 'neg_11.25_to_about_9.25', 'neg_8.25_to_6.25']
for sweep_param in sweep_params:

    if test_dataset == 'synthetic_experiment':
        """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution, particle_density=None,
                                          single_particle_calibration=True, static_templates=False,
                                          hard_baseline=True, particles_overlapping=False, sweep_method=sweep_method,
                                          sweep_param=sweep_param, use_stack_id=None).unpack()"""
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=None,
                                         number_of_images=num_test_images,
                                         particle_distribution=particle_distribution, particle_density=sweep_param,
                                         single_particle_calibration=False, static_templates=True,
                                         hard_baseline=True, particles_overlapping=False, sweep_method=sweep_method,
                                         sweep_param=sweep_param, use_stack_id=None).unpack()
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
                                          single_particle_calibration=False, static_templates=True,
                                          hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                          sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                         number_of_images=num_test_images,
                                         particle_distribution=particle_distribution, particle_density=sweep_param,
                                         single_particle_calibration=False, static_templates=True,
                                         hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                         sweep_param=sweep_param).unpack()
    elif test_dataset == '08.02.21 - bpe.g2 deflection':
        """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, particle_distribution=particle_distribution,
                                          sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                         particles_overlapping=False, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param,
                                         use_stack_id=10).unpack()
    elif test_dataset == '02.07.22_membrane_characterization':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution, particle_density=None,
                                          single_particle_calibration=False, static_templates=True,
                                          hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                          sweep_param=sweep_param, use_stack_id=None).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=None).unpack()
    elif test_dataset == 'zipper':
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=False, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=sweep_param).unpack()
    elif test_dataset == '11.06.21_z-micrometer-v2':
        """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, particle_distribution=particle_distribution,
                                          sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                         static_templates=True, single_particle_calibration=True, hard_baseline=True,
                                         particles_overlapping=True, particle_distribution=particle_distribution,
                                         sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=sweep_param).unpack()
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
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    elif test_dataset == '11.02.21-BPE_Pressure_Deflection_20X':
        calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    elif test_dataset == '20X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
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
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
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
                                          static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                          particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                         particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
    else:
        raise ValueError("No dataset found.")

    # ----- ----- ----- ----- TEST ----- ----- ----- ----- ----- ----- -----
    GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None, return_variables=None)

# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- -----
"""sweep_params = ['_']
for sweep_param in sweep_params:
    calib_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution, particle_density=sweep_param,
                                      single_particle_calibration=False, static_templates=True,
                                      hard_baseline=True, particles_overlapping=False, sweep_method=sweep_method,
                                      sweep_param=sweep_param, use_stack_id=None).unpack()
    test_settings2 = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=None,
                                     number_of_images=num_test_images,
                                     particle_distribution=particle_distribution, particle_density=sweep_param,
                                     single_particle_calibration=False, static_templates=True,
                                     hard_baseline=True, particles_overlapping=False, sweep_method=sweep_method,
                                     sweep_param=sweep_param, use_stack_id=None).unpack()
    GdpytCharacterize.test(calib_settings2, test_settings2, return_variables=None)
"""
# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- -----


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