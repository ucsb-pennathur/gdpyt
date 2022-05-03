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

# outdated parameters
calib_col, calib_set = None, None
known_z, num_test_images, nl = None, 470, 2

# ----- ----- ----- ----- TEST DATASET UNPACKER ----- ----- ----- ----- ----- ----- -----

test_dataset = '11.02.21-BPE_Pressure_Deflection_20X'  # '10.07.21-BPE_Pressure_Deflection'
particle_distribution = 'off-bpe'
sweep_method = 'spct'
sweep_params = [['calib2', 'calib3']]

# generate calibration collection and calibration set
calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                  particle_distribution=particle_distribution,
                                  single_particle_calibration=True,
                                  static_templates=False,
                                  hard_baseline=False,
                                  particles_overlapping=False,
                                  sweep_method=sweep_method,
                                  sweep_param=sweep_params[0],
                                  use_stack_id='best').unpack()
calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')


# GDPyT test collection

pp = True
if pp:
    for sweep_param in sweep_params:
        if test_dataset == '10.07.21-BPE_Pressure_Deflection':
            calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                              particle_distribution=particle_distribution,
                                              single_particle_calibration=True, static_templates=False,
                                              hard_baseline=False, particles_overlapping=False, sweep_method=sweep_method,
                                              sweep_param=sweep_param, use_stack_id='best').unpack()
            test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                             static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                             particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                             use_stack_id='best', particle_distribution=particle_distribution).unpack()
        elif test_dataset == '11.02.21-BPE_Pressure_Deflection_20X':
            """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                              static_templates=True,
                                              single_particle_calibration=False,
                                              hard_baseline=True,
                                              particles_overlapping=True,
                                              sweep_method=sweep_method,
                                              sweep_param=sweep_param).unpack()"""
            test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                             particle_distribution=particle_distribution,
                                             single_particle_calibration=True,
                                             static_templates=False,
                                             hard_baseline=False,
                                             particles_overlapping=False,
                                             sweep_method=sweep_method,
                                             sweep_param=sweep_param,
                                             use_stack_id='best').unpack()
        elif test_dataset == '11.06.21_z-micrometer-v2':
            calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                              static_templates=False, single_particle_calibration=True,
                                              hard_baseline=False,
                                              particles_overlapping=False,
                                              particle_distribution=particle_distribution,
                                              sweep_method=sweep_method, sweep_param=sweep_param).unpack()
            test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                             static_templates=False, single_particle_calibration=True,
                                             hard_baseline=False,
                                             particles_overlapping=False,
                                             particle_distribution=particle_distribution,
                                             sweep_method=sweep_method, sweep_param=sweep_param,
                                             use_stack_id=44).unpack()
        elif test_dataset == 'synthetic_overlap_noise-level':
            calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=nl,
                                              number_of_images=num_test_images,
                                              particle_distribution=particle_distribution,
                                              particle_density=sweep_param,
                                              single_particle_calibration=False, static_templates=True,
                                              hard_baseline=True, particles_overlapping=False,
                                              sweep_method=sweep_method,
                                              sweep_param=sweep_param).unpack()
            test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                             number_of_images=num_test_images,
                                             particle_distribution=particle_distribution,
                                             particle_density=sweep_param,
                                             single_particle_calibration=False, static_templates=True,
                                             hard_baseline=True, particles_overlapping=False,
                                             sweep_method=sweep_method, sweep_param=sweep_param,
                                             use_stack_id=None).unpack()
        elif test_dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':
            test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                             static_templates=False, single_particle_calibration=True,
                                             hard_baseline=False,
                                             particles_overlapping=False,
                                             particle_distribution=particle_distribution,
                                             sweep_method=sweep_method, sweep_param=sweep_param,
                                             use_stack_id='best').unpack()

        elif pp == 'others':
            if test_dataset == 'synthetic_experiment':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                                  number_of_images=num_test_images,
                                                  particle_distribution=particle_distribution, particle_density=None,
                                                  single_particle_calibration=True, static_templates=False,
                                                  hard_baseline=True, particles_overlapping=False, sweep_method=sweep_method,
                                                  sweep_param=sweep_param, use_stack_id=None).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=nl,
                                                 number_of_images=num_test_images,
                                                 particle_distribution=particle_distribution,
                                                 particle_density=sweep_param,
                                                 single_particle_calibration=True, static_templates=False,
                                                 hard_baseline=False, particles_overlapping=False,
                                                 sweep_method=sweep_method,
                                                 sweep_param=sweep_param, use_stack_id=None).unpack()
            elif test_dataset == '02.07.22_membrane_characterization':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=None,
                                                  number_of_images=num_test_images,
                                                  particle_distribution=particle_distribution, particle_density=None,
                                                  single_particle_calibration=False, static_templates=True,
                                                  hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                                  sweep_param=sweep_param, use_stack_id=None).unpack()"""
                """test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                 particles_overlapping=True, particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=None).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=False, single_particle_calibration=True,
                                                 hard_baseline=False,
                                                 particles_overlapping=False, sweep_method=sweep_method,
                                                 sweep_param=sweep_param,
                                                 use_stack_id=19, particle_distribution=particle_distribution).unpack()
            elif test_dataset == '02.06.22_membrane_characterization':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                              particle_distribution=particle_distribution,
                                              single_particle_calibration=False, static_templates=True,
                                              hard_baseline=True, particles_overlapping=True, sweep_method=sweep_method,
                                              sweep_param=sweep_param, use_stack_id=None).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=True, single_particle_calibration=False,
                                                 hard_baseline=True,
                                                 particles_overlapping=True,
                                                 particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=None).unpack()
            elif test_dataset == '11.06.21_z-micrometer-v2':
                calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=False, single_particle_calibration=True,
                                                  hard_baseline=False,
                                                  particles_overlapping=False,
                                                  particle_distribution=particle_distribution,
                                                  sweep_method=sweep_method, sweep_param=sweep_param).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True,
                                                 hard_baseline=False,
                                                 particles_overlapping=False,
                                                 particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id='best').unpack()
            elif test_dataset == 'JP-EXF01-20':
                calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration', noise_level=2,
                                                  number_of_images=50, particle_distribution='Dataset_I',
                                                  particle_density='361', static_templates=False, hard_baseline=False,
                                                  single_particle_calibration=True,
                                                  particles_overlapping=False).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test', noise_level=2,
                                                 number_of_images=num_test_images, particle_distribution='Dataset_I',
                                                 particle_density='361', static_templates=False, hard_baseline=False,
                                                 single_particle_calibration=True, particles_overlapping=False).unpack()
            elif test_dataset == '08.02.21 - bpe.g2 deflection':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, particle_distribution=particle_distribution,
                                                  sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=False, single_particle_calibration=True,
                                                 hard_baseline=False,
                                                 particles_overlapping=False,
                                                 particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=10).unpack()
            elif test_dataset == '11.09.21_z-micrometer-v3':
                calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, particle_distribution=particle_distribution,
                                                  sweep_method=sweep_method, sweep_param=sweep_param).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                 particles_overlapping=True, particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param, known_z=known_z).unpack()
            elif test_dataset == 'zipper':
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=True, single_particle_calibration=False,
                                                 hard_baseline=True,
                                                 particles_overlapping=True,
                                                 particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=sweep_param).unpack()
            elif test_dataset == '20X_1Xmag_0.87umNR':
                calib_settings = dataset_unpacker(dataset=test_dataset,
                                                  collection_type='calibration',
                                                  static_templates=True,
                                                  single_particle_calibration=False,
                                                  hard_baseline=True,
                                                  particles_overlapping=True,
                                                  sweep_method=sweep_method,
                                                  sweep_param=sweep_param,
                                                  use_stack_id=None,
                                                  ).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset,
                                                 collection_type='meta-test',
                                                 static_templates=True,
                                                 single_particle_calibration=False,
                                                 hard_baseline=True,
                                                 particles_overlapping=True,
                                                 sweep_method=sweep_method,
                                                 sweep_param=sweep_param,
                                                 use_stack_id=None,
                                                 ).unpack()
            elif test_dataset == '20X_1Xmag_5.61umPink_HighInt':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=sweep_param).unpack()
            elif test_dataset == '11.02.21-BPE_Pressure_Deflection_20X':
                calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                 particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
            elif test_dataset == '20X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=sweep_param).unpack()
            elif test_dataset == '10X_0.5Xmag_2.15umNR_HighInt_0.03XHg':
                calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                 static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                 particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()
            elif test_dataset == '10X_1Xmag_5.1umNR_HighInt_0.06XHg':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=True, single_particle_calibration=False, hard_baseline=True,
                                                  particles_overlapping=True, sweep_method=sweep_method, sweep_param=sweep_param).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=sweep_param).unpack()
            elif test_dataset == '20X_1Xmag_0.87umNR_on_Silpuran':
                """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                  static_templates=False, single_particle_calibration=False,
                                                  hard_baseline=True).unpack()"""
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                                 use_stack_id=sweep_param).unpack()
            elif test_dataset == '20X_1Xmag_2.15umNR_HighInt_0.03XHg':
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=sweep_param).unpack()
            elif test_dataset == '20X_1Xmag_5.1umNR_HighInt_0.03XHg':
                test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                 static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                 particles_overlapping=False, particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=sweep_param).unpack()

        else:
            raise ValueError("No dataset found.")

        # ----- ----- ----- ----- TEST ----- ----- ----- ----- ----- ----- -----
        if calib_col is not None:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=calib_col, calib_set=calib_set,
                                   return_variables=None)
        else:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None,
                                   return_variables=None)

# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- ----- ----- ----

# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- ----- ----- ----
multi = False
if multi:

    calib_col, calib_set = None, None
    known_z = None
    num_test_images = 10
    nl = 1

    test_dataset = '20X_1Xmag_0.87umNR'  #
    particle_distribution = 'glass'
    sweep_method = 'meta'  # 'micrometer_5um'
    sweep_params = ['spct']

    # generate calibration collection and calibration set
    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                      particle_distribution=particle_distribution,
                                      single_particle_calibration=True, static_templates=False,
                                      hard_baseline=False, particles_overlapping=False, sweep_method=sweep_method,
                                      sweep_param='spct-cal', use_stack_id='best').unpack()
    calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')

    for sweep_param in sweep_params:

        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                         static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                         particles_overlapping=False, sweep_method=sweep_method, sweep_param=sweep_param,
                                         use_stack_id=sweep_param, particle_distribution=particle_distribution).unpack()

        if calib_col is not None:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=calib_col, calib_set=calib_set,
                                   return_variables=None)
        else:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None, return_variables=None)



print('Analysis completed without errors.')