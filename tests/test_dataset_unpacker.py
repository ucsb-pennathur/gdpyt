"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytImageCollection, GdpytSetup, GdpytCharacterize
from gdpyt.utils.datasets import dataset_unpacker
"""
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.morphology import disk, square
"""

# outdated parameters
calib_col, calib_set = None, None
known_z, num_test_images, nl = None, None, None  # None, 500, 15
return_variables = None

# ----- ----- ----- ----- TEST DATASET UNPACKER ----- ----- ----- ----- ----- ----- -----

for test_dataset in ['w25_a1_5pT_test1']:

    idpt_publication = False
    if idpt_publication:

        test_dataset = '11.06.21_z-micrometer-v2'  # '02.06.22_membrane_characterization'  # '11.06.21_z-micrometer-v2'

        m = 'spct'  # 'idpt'  # 'spct'

        if test_dataset == '02.06.22_membrane_characterization':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                            ['dynamic_neg_first', 11, 1],
                            # ['dynamic_neg_first', 13, 1],
                            # ['dynamic_neg_first', 11, 10],
                            # ['dynamic_neg_first', 13, 1],
                            ]  # [40, 42, 44, 35, 0, 78]
            calib_stack_id = 34  # 'nearest'

        elif test_dataset == '11.06.21_z-micrometer-v2':
            particle_distribution = 'SILPURAN'
            sweep_method = 'dzc'  # 'micrometer_5um'
            sweep_params = [1]  #
            calib_stack_id = 44  # 'nearest'
            return_variables = 'calibration'
            test_settings = None
        else:
            raise ValueError()

        """ elif test_dataset == 'synthetic_overlap_noise-level':
            particle_distribution = 'single'
            sweep_method = 'rescale3'
            sweep_params = ['xy38', 'xy37.5', 'xy37.75', 'xy37.6', 'x37.5_y37.6', 'x37.5_y37.75', 'x37.5_y38']
            calib_stack_id = None  # 'nearest'
        
        elif test_dataset == 'synthetic_example':
            particle_distribution = 'rand'
            sweep_method = 'rand'
            sweep_params = [1]
            calib_stack_id = None  # 'nearest'
        
        elif test_dataset == '20X_1Xmag_0.87umNR':
            particle_distribution = 'Glass'
            sweep_method = 'meta'
            sweep_params = [1]
            calib_stack_id = 43  # 'nearest'
        
        elif test_dataset == '20X_1Xmag_2.15umNR_HighInt_0.03XHg':
            particle_distribution = 'glass'
            sweep_method = 'meta'
            sweep_params = [1]
            calib_stack_id = 5  # 'nearest'
        
        elif test_dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':
            particle_distribution = 'glass'
            sweep_method = 'meta'
            sweep_params = [1]
            calib_stack_id = 34  # 'nearest'
        """

        if m == 'spct':
            single_particle_calibration = True
            static_templates = False
            hard_baseline = False
            particles_overlapping = False
        else:
            single_particle_calibration = False
            static_templates = True
            hard_baseline = True
            particles_overlapping = False


        # generate calibration collection and calibration set
        """calib_settings = dataset_unpacker(dataset=test_dataset,
                                          collection_type='calibration',
                                          noise_level=nl,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution,
                                          particle_density=None,
                                          single_particle_calibration=single_particle_calibration,
                                          static_templates=static_templates,
                                          hard_baseline=hard_baseline,
                                          particles_overlapping=particles_overlapping,
                                          sweep_method=sweep_method,
                                          sweep_param='cal').unpack()
        calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')"""

    else:

        # test_dataset = ['wA_c1_test1', 'wA_a1_test1']

        if test_dataset == 'wA2':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                            ['50V', 'n=1'], ['-50V', 'n=1'],
                            ['55V', 'n=1'], ['-55V', 'n=1'],
                            ['60V', 'n=1'], ['-60V', 'n=1'],
                            ['65V', 'n=1'], ['-65V', 'n=1'],
                            ['70V', 'n=1'], ['-70V', 'n=1'],
                            ['75V', 'n=1'], ['-75V', 'n=1'],
                            ['75V', 'n=2'], ['-75V', 'n=2'],
                            ['75V', 'n=3'], ['-75V', 'n=3'],
                            ['80V', 'n=1'], ['-80V', 'n=1'],
                            ['80V', 'n=2'], ['-80V', 'n=2'],
                            ['80V', 'n=3'], ['-80V', 'n=3'],
                            ['100V', 'n=1'], ['-100V', 'n=1'],
                            ['125V', 'n=1'], ['-125V', 'n=1'],
                            ['125V', 'n=2'], ['-125V', 'n=2'],
                            ['125V', 'n=3'], ['-125V', 'n=3'],
                            ['25V', 'n=1'], ['-25V', 'n=1'],
                            ['0V', 'n=1'],
                            ['0V', 'n=2'],
                            ]
        elif test_dataset == 'wA_b1_test2':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params1 = [
                ['50V', 'n=1'], ['-50V', 'n=1'],
                ['200V', 'n=1'], ['-200V', 'n=1'],
                ['300V', 'n=1'], ['-300V', 'n=1'],
                ['400V', 'n=1'], ['-400V', 'n=1'],
                ['500V', 'n=1'], ['-500V', 'n=1'],
                ['600V', 'n=1'], ['-600V', 'n=1'],
                ['650V', 'n=1'], ['-650V', 'n=1'],
                ['700V', 'n=1'], ['-700V', 'n=1'],
                ['750V', 'n=1'], ['-750V', 'n=1'],
                ['800V', 'n=1'], ['-800V', 'n=1'],
                ['850V', 'n=1'], ['-850V', 'n=1'],
                ['900V', 'n=1'], ['-900V', 'n=1'],
                ['950V', 'n=1'], ['-950V', 'n=1'],
                ['1000V', 'n=1'], ['-1000V', 'n=1'],
                ['1050V', 'n=1'], ['-1050V', 'n=1'],
                ['1100V', 'n=1'], ['-1100V', 'n=1'],
                ['1200V', 'n=1'], ['-1200V', 'n=1'],
                ['1250V', 'n=1'], ['-1250V', 'n=1'],
                ['1300V', 'n=1'], ['-1300V', 'n=1'],
                ['1350V', 'n=1'], ['-1350V', 'n=1'],
                ['1400V', 'n=1'], ['-1400V', 'n=1'],
                ['1450V', 'n=1'], ['-1450V', 'n=1'],
                ['1500V', 'n=1'], ['1500V', 'n=2'], ['1500V', 'n=3'], ['-1500V', 'n=1'], ['-1500V', 'n=2'], ['-1500V', 'n=3'],
                ['1600V', 'n=1'], ['-1600V', 'n=1'],
                ['1700V', 'n=1'], ['-1700V', 'n=1'],
                ['1800V', 'n=1'], ['-1800V', 'n=1'],
                ['1900V', 'n=1'], ['-1900V', 'n=1'],
                ['2000V', 'n=1'], ['-2000V', 'n=1'],
                ['2100V', 'n=1'], ['2100V', 'n=3'], ['-2100V', 'n=1'], ['-2100V', 'n=2'], ['-2100V', 'n=3'],
                ['0V', 'n=1'],
                            ]

            sweep_params2 = [
                ['2100V', 'n=1'], ['-2100V', 'n=1'],
                ['2200V', 'n=1'], ['-2200V', 'n=1'],
                ['2300V', 'n=1'], ['2300V', 'n=3'], ['-2300V', 'n=1'], ['-2300V', 'n=2'], ['-2300V', 'n=3'], ['-2300V', 'n=3'],
            ]

            sweep_params = sweep_params2
        elif test_dataset == 'w18_b1_test3':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                ['2500V_10ms_500ms'],
                ['2500V_15ms_750ms'],
                ['2500V_20ms_1000ms'],
                ['2500V_25ms_1250ms'],
                ['2500V_30ms_1500ms'],
                ['2500V_35ms_1750ms'],
                ['2500V_40ms_2000ms'],
                ['2500V_45ms_2250ms'],
                ['2500V_50ms_2500ms'],
                ['2000V_20msCycle'],
                ['2000V_25msCycle'],
                ['2000V_30msCycle'],
                ['2500V_55ms_2750ms'],
                ['2500V_60ms_3000ms'],
                ['2500V_65ms_3250ms'],
                ['2500V_70ms_3500ms'],
                ['2500V_75ms_3750ms'],
                ['2500V_80ms_4000ms'],
                ['2000V_30msCycle'],
                            ]
        elif test_dataset == 'wA_a1_test1':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                ['300V', 'n=1'], ['-300V', 'n=1'],
                ['400V', 'n=1'], ['-400V', 'n=1'],
                ['500V', 'n=1'], ['-500V', 'n=1'],
                ['600V', 'n=1'], ['-600V', 'n=1'], ['600V', 'n=2'], ['-600V', 'n=2'],
                ['700V', 'n=1'], ['-700V', 'n=1'],
                ['800V', 'n=1'], ['-800V', 'n=1'],
                ['900V', 'n=1'], ['-900V', 'n=1'],
                ['1000V', 'n=1'], ['-1000V', 'n=1'], ['1000V', 'n=2'], ['-1000V', 'n=2'],
                ['1100V', 'n=1'], ['-1100V', 'n=1'],
                ['1200V', 'n=1'], ['-1200V', 'n=1'],
                ['1300V', 'n=1'], ['-1300V', 'n=1'],
                ['1400V', 'n=1'], ['-1400V', 'n=1'],
                ['1500V', 'n=1'], ['-1500V', 'n=1'],
                ['1600V', 'n=1'], ['-1600V', 'n=1'],
                ['1700V', 'n=1'], ['-1700V', 'n=1'],
                ['1800V', 'n=1'], ['-1800V', 'n=1'],
                ['1900V', 'n=1'], ['-1900V', 'n=1'],
                ['2000V', 'n=1'], ['-2000V', 'n=1'],
                ['2200V', 'n=1'], ['-2200V', 'n=1'],
                ['2100V', 'n=1'], ['-2100V', 'n=1'],
                ['2300V', 'n=1'], ['-2300V', 'n=1'], ['2300V', 'n=2'], ['-2300V', 'n=2'], ['2300V', 'n=3'], ['-2300V', 'n=3'],
                ['2400V', 'n=1'], ['-2400V', 'n=1'],
                ['2500V', 'n=1'], ['-2500V', 'n=1'],
                ['2600V', 'n=1'], ['-2600V', 'n=1'],
                ['2700V', 'n=1'], ['-2700V', 'n=1'], ['2700V', 'n=2'], ['-2700V', 'n=2'], ['2700V', 'n=3'], ['-2700V', 'n=3'],
                ['0V', 'n=1'],
                            ]
        elif test_dataset == 'wA_c1_test1':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                ['300V', 'n=1'], ['-300V', 'n=1'],
                ['600V', 'n=1'], ['-600V', 'n=1'],
                ['700V', 'n=1'],  # ['-700V', 'n=1'],
                ['800V', 'n=1'], ['-800V', 'n=1'],
                ['900V', 'n=1'], ['-900V', 'n=1'],
                ['1000V', 'n=1'], ['-1000V', 'n=1'],
                ['1100V', 'n=1'], ['-1100V', 'n=1'],
                ['1200V', 'n=1'], ['-1200V', 'n=1'],
                ['1300V', 'n=1'], ['-1300V', 'n=1'],
                ['1400V', 'n=1'], ['-1400V', 'n=1'],
                ['1500V', 'n=1'], ['-1500V', 'n=1'],
                ['1600V', 'n=1'], ['-1600V', 'n=1'],
                ['1700V', 'n=1'], ['-1700V', 'n=1'],
                ['1800V', 'n=1'], ['-1800V', 'n=1'],
                ['1900V', 'n=1'], ['-1900V', 'n=1'],
                ['2000V', 'n=1'], ['-2000V', 'n=1'],
                ['2100V', 'n=1'], ['-2100V', 'n=1'],
                ['2200V', 'n=1'], ['-2200V', 'n=1'],
                ['2300V', 'n=1'], ['-2300V', 'n=1'],
                ['2400V', 'n=1'], ['-2400V', 'n=1'],
                ['2500V', 'n=1'], ['-2500V', 'n=1'],
                ['2600V', 'n=1'], ['-2600V', 'n=1'],
                ['0V', 'n=1'], ['0V', 'n=2'],
                            ]
        elif test_dataset == 'w18_c1_test3':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params1 = [
                ['2500V_10ms_500ms'],
                ['2500V_20ms_1000ms'],
                ['2500V_30ms_1500ms'],
                ['2500V_40ms_2000ms'],
                ['2500V_50ms_2500ms'],
                ['2500V_60ms_3000ms'],
                ['2500V_70ms_3500ms'],
                ['2500V_80ms_4000ms'],
                            ]
            sweep_params = [
                ['2500V_5sRamp_n=1'],
                ['2500V_5sRamp_n=2'],
                ['2500V_5sRamp_n=3'],
                ['2500V_7sRamp_n=1'],
                ['2500V_7sRamp_n=2'],
                ['2500V_7sRamp_n=3'],
                            ]
        elif test_dataset == 'w18_c1_5pT_test2':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                # ['200V_6sRamp_n=1'], ['-200V_6sRamp_n=1'],
                # ['200V_8sRamp_n=1'], ['-200V_8sRamp_n=1'],
                # ['250V_4sRamp_n=1'], ['-250V_4sRamp_n=1'],
                # ['250V_6sRamp_n=1'], ['-250V_6sRamp_n=1'],
                # ['250V_8sRamp_n=1'], ['-250V_8sRamp_n=1'],
                # ['300V_8sRamp_n=1'], ['-300V_8sRamp_n=1'],
                # ['350V_8sRamp_n=1'], ['-350V_8sRamp_n=1'],
                # ['400V_8sRamp_n=1'], ['-400V_8sRamp_n=1'],
                # ['200V_2sRamp_n=1'], require "regular" parameters (i.e., baseline image = best focus)
                # ['-200V_2sRamp_n=1'],require "regular" parameters (i.e., baseline image = best focus)
                # the below tests begin defocused so require special parameters
                # ['375V_2sRamp_n=1'],
                # ['-375V_2sRamp_n=1'],
                # ['175V_2sRamp_n=1'],
                # ['-175V_2sRamp_n=1'],
                # ['225V_2sRamp_n=1'], ['225V_2sRamp_n=2'],
                # ['-225V_2sRamp_n=1'], ['-225V_2sRamp_n=2'],
                # ['250V_2sRamp_n=1'],
                # ['-250V_2sRamp_n=1'],
                # ['275V_2sRamp_n=1'],
                # ['-275V_2sRamp_n=1'],
                # ['300V_2sRamp_n=1'],
                # ['-300V_2sRamp_n=1'],
                # ['325V_2sRamp_n=1'],
                # ['-325V_2sRamp_n=1'],
                # ['350V_2sRamp_n=1'],
                # ['-350V_2sRamp_n=1'],
                # ['0V_tick47'],
                # ['200V_2sRamp_n=1'], ['-200V_2sRamp_n=1'],
                # the below tests are "...fin_tests..."
                ['100V_250ms'], ['100V_250ms_n=2'], ['-100V_250ms_n=2'],
                ['0V_preProbe'], ['0V_postProbe'],
                ['50V_250ms'], ['-50V_250ms'],
                ['75V_250ms'], ['-75V_250ms'],
                ['200V_250ms'], ['-200V_250ms'],
                ['225V_250ms'], ['-225V_250ms'],
                ['250V_250ms'], ['-250V_250ms'],
                ['275V_250ms'], ['-275V_250ms'],
                ['300V_250ms'], ['-300V_250ms'], ['300V_250ms_n=2_stopHSV'], ['-300V_250ms_n=2_stopHSV'],
                ['325V_250ms_n=1'], ['-325V_250ms_n=1'],
            ]
        elif test_dataset == 'w18_c1_0pT_test1':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = ['400V_7sRamp_n=1',]
        elif test_dataset == 'w25_a1_5pT_test1':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = ['100V_4sRamp_n=1', '0V_ztick40_afterProbeContact_n=2']
        elif test_dataset == 'wA_c1_test2':
            particle_distribution = '10X'
            sweep_method = 'testset'
            sweep_params = [
                ['2000V', 'n=1'], ['-2000V', 'n=1'], ['2000V', 'n=2'], ['-2000V', 'n=2'],
                ['2100V', 'n=1'], ['-2100V', 'n=1'],
                ['2200V', 'n=1'], ['-2200V', 'n=1'],
                ['2300V', 'n=1'], ['-2300V', 'n=1'],
                ['1600V', 'n=1'], ['-1600V', 'n=1'], ['1600V', 'n=2'], ['-1600V', 'n=2'],
                ['1700V', 'n=1'], ['-1700V', 'n=1'], ['1700V', 'n=2'], ['-1700V', 'n=2'],
                ['1800V', 'n=1'], ['-1800V', 'n=1'], ['1800V', 'n=2'], ['-1800V', 'n=2'],
                ['1900V', 'n=1'], ['-1900V', 'n=1'], ['1900V', 'n=2'], ['-1900V', 'n=2'],
                ['1100V', 'n=1'], ['-1100V', 'n=1'], ['1100V', 'n=2'], ['-1100V', 'n=2'],
                ['1300V', 'n=1'], ['-1300V', 'n=1'], ['1300V', 'n=2'], ['-1300V', 'n=2'],
                ['1500V', 'n=1'], ['-1500V', 'n=1'], ['1500V', 'n=2'], ['-1500V', 'n=2'],
                ['900V', 'n=1'], ['-900V', 'n=1'], ['900V', 'n=2'], ['-900V', 'n=2'],
                ['700V', 'n=1'], ['-700V', 'n=1'], ['700V', 'n=2'],  ['-700V', 'n=2'],
                ['500V', 'n=1'], ['-500V', 'n=1'],
                ['0V', 'n=1'],
                            ]

        else:
            raise ValueError("Dataset not found.")

        single_particle_calibration = False
        static_templates = True
        hard_baseline = True
        particles_overlapping = False
        num_test_images = None
        calib_stack_id = None

        # generate calibration collection and calibration set
        calib_settings = dataset_unpacker(dataset=test_dataset,
                                          collection_type='calibration',
                                          noise_level=nl,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution,
                                          particle_density=None,
                                          single_particle_calibration=single_particle_calibration,
                                          static_templates=static_templates,
                                          hard_baseline=hard_baseline,
                                          particles_overlapping=particles_overlapping,
                                          sweep_method=sweep_method,
                                          sweep_param='cal').unpack()
        calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')

    # GDPyT test collection

    pp = True
    if pp:
        pp = 'others'
        for sweep_param in sweep_params:
            if idpt_publication:
                if test_dataset == '10.07.21-BPE_Pressure_Deflection':
                    """calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                      particle_distribution=particle_distribution,
                                                      single_particle_calibration=True, static_templates=False,
                                                      hard_baseline=False, particles_overlapping=False, sweep_method=sweep_method,
                                                      sweep_param=sweep_param, use_stack_id='best').unpack()"""
                    test_settings = dataset_unpacker(dataset=test_dataset,
                                                     collection_type='test',
                                                     static_templates=static_templates,
                                                     single_particle_calibration=single_particle_calibration,
                                                     hard_baseline=hard_baseline,
                                                     particles_overlapping=particles_overlapping,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id,
                                                     particle_distribution=particle_distribution).unpack()
                elif test_dataset == '02.06.22_membrane_characterization':
                    calib_settings = dataset_unpacker(dataset=test_dataset,
                                                      collection_type='calibration',
                                                      noise_level=nl,
                                                      number_of_images=num_test_images,
                                                      particle_distribution=particle_distribution,
                                                      particle_density=None,
                                                      single_particle_calibration=single_particle_calibration,
                                                      static_templates=static_templates,
                                                      hard_baseline=hard_baseline,
                                                      particles_overlapping=particles_overlapping,
                                                      use_stack_id=calib_stack_id,
                                                      sweep_method=sweep_method,
                                                      sweep_param=sweep_param).unpack()
                    test_settings = dataset_unpacker(dataset=test_dataset,
                                                     collection_type='test',
                                                     particle_distribution=particle_distribution,
                                                     single_particle_calibration=single_particle_calibration,
                                                     static_templates=static_templates,
                                                     hard_baseline=hard_baseline,
                                                     particles_overlapping=particles_overlapping,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id).unpack()
                elif test_dataset == '11.02.21-BPE_Pressure_Deflection_20X':
                    calib_settings = dataset_unpacker(dataset=test_dataset, collection_type='calibration',
                                                      static_templates=True,
                                                      single_particle_calibration=False,
                                                      hard_baseline=True,
                                                      particles_overlapping=True,
                                                      sweep_method=sweep_method,
                                                      sweep_param=sweep_param,
                                                      use_stack_id=calib_stack_id).unpack()
                    test_settings = dataset_unpacker(dataset=test_dataset, collection_type='test',
                                                     particle_distribution=particle_distribution,
                                                     single_particle_calibration=True,
                                                     static_templates=False,
                                                     hard_baseline=False,
                                                     particles_overlapping=False,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id).unpack()
                elif test_dataset == '11.06.21_z-micrometer-v2':
                    calib_settings = dataset_unpacker(dataset=test_dataset,
                                                      collection_type='calibration',
                                                      static_templates=static_templates,
                                                      single_particle_calibration=single_particle_calibration,
                                                      hard_baseline=hard_baseline,
                                                      particles_overlapping=particles_overlapping,
                                                      particle_distribution=particle_distribution,
                                                      sweep_method=sweep_method,
                                                      sweep_param=sweep_param,
                                                      use_stack_id=calib_stack_id).unpack()
                    """test_settings = dataset_unpacker(dataset=test_dataset,
                                                     collection_type='test',
                                                     static_templates=static_templates,
                                                     single_particle_calibration=single_particle_calibration,
                                                     hard_baseline=hard_baseline,
                                                     particles_overlapping=particles_overlapping,
                                                     particle_distribution=particle_distribution,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id).unpack()"""
                elif test_dataset == 'synthetic_overlap_noise-level':
                    """calib_settings = dataset_unpacker(dataset=test_dataset,
                                                      collection_type='calibration',
                                                      noise_level=nl,
                                                      number_of_images=num_test_images,
                                                      particle_distribution=particle_distribution,
                                                      particle_density=sweep_param,
                                                      single_particle_calibration=single_particle_calibration,
                                                      static_templates=static_templates,
                                                      hard_baseline=hard_baseline,
                                                      particles_overlapping=particles_overlapping,
                                                      sweep_method=sweep_method,
                                                      sweep_param=sweep_param).unpack()"""
                    test_settings = dataset_unpacker(dataset=test_dataset,
                                                     collection_type='test',
                                                     noise_level=nl,
                                                     number_of_images=num_test_images,
                                                     particle_distribution=particle_distribution,
                                                     particle_density=sweep_param,
                                                     single_particle_calibration=single_particle_calibration,
                                                     static_templates=static_templates,
                                                     hard_baseline=hard_baseline,
                                                     particles_overlapping=particles_overlapping,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id).unpack()
                elif test_dataset == '10X_1Xmag_2.15umNR_HighInt_0.12XHg':
                    calib_settings = dataset_unpacker(dataset=test_dataset,
                                                      collection_type='calibration',
                                                      particle_distribution=particle_distribution,
                                                      static_templates=static_templates,
                                                      single_particle_calibration=single_particle_calibration,
                                                      hard_baseline=hard_baseline,
                                                      particles_overlapping=particles_overlapping,
                                                      sweep_method=sweep_method,
                                                      sweep_param=sweep_param,
                                                      use_stack_id=calib_stack_id,
                                                      ).unpack()
                    test_settings = dataset_unpacker(dataset=test_dataset,
                                                     collection_type='test',
                                                     particle_distribution=particle_distribution,
                                                     static_templates=static_templates,
                                                     single_particle_calibration=single_particle_calibration,
                                                     hard_baseline=hard_baseline,
                                                     particles_overlapping=particles_overlapping,
                                                     sweep_method=sweep_method,
                                                     sweep_param=sweep_param,
                                                     use_stack_id=calib_stack_id,
                                                     ).unpack()

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
                    elif test_dataset == 'synthetic_example':
                        calib_settings = dataset_unpacker(dataset=test_dataset,
                                                          collection_type='calibration',
                                                          noise_level=nl,
                                                          number_of_images=num_test_images,
                                                          particle_distribution=particle_distribution,
                                                          particle_density=sweep_param,
                                                          single_particle_calibration=single_particle_calibration,
                                                          static_templates=static_templates,
                                                          hard_baseline=hard_baseline,
                                                          particles_overlapping=particles_overlapping,
                                                          sweep_method=sweep_method,
                                                          sweep_param=sweep_param).unpack()
                        test_settings = dataset_unpacker(dataset=test_dataset,
                                                         collection_type='test',
                                                         noise_level=nl,
                                                         number_of_images=num_test_images,
                                                         particle_distribution=particle_distribution,
                                                         particle_density=sweep_param,
                                                         single_particle_calibration=single_particle_calibration,
                                                         static_templates=static_templates,
                                                         hard_baseline=hard_baseline,
                                                         particles_overlapping=particles_overlapping,
                                                         sweep_method=sweep_method,
                                                         sweep_param=sweep_param,
                                                         use_stack_id=calib_stack_id).unpack()
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
                                                          static_templates=static_templates,
                                                          single_particle_calibration=single_particle_calibration,
                                                          hard_baseline=hard_baseline,
                                                          particles_overlapping=particles_overlapping,
                                                          sweep_method=sweep_method,
                                                          sweep_param=sweep_param,
                                                          use_stack_id=calib_stack_id,
                                                          ).unpack()
                        test_settings = dataset_unpacker(dataset=test_dataset,
                                                         collection_type='meta-test',
                                                         static_templates=static_templates,
                                                         single_particle_calibration=single_particle_calibration,
                                                         hard_baseline=hard_baseline,
                                                         particles_overlapping=particles_overlapping,
                                                         sweep_method=sweep_method,
                                                         sweep_param=sweep_param,
                                                         use_stack_id=calib_stack_id,
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
                        calib_settings = dataset_unpacker(dataset=test_dataset,
                                                          collection_type='calibration',
                                                          particle_distribution=particle_distribution,
                                                          static_templates=static_templates,
                                                          single_particle_calibration=single_particle_calibration,
                                                          hard_baseline=hard_baseline,
                                                          particles_overlapping=particles_overlapping,
                                                          sweep_method=sweep_method,
                                                          sweep_param=sweep_param,
                                                          use_stack_id=calib_stack_id,
                                                          ).unpack()
                        test_settings = dataset_unpacker(dataset=test_dataset,
                                                         collection_type='meta-test',
                                                         particle_distribution=particle_distribution,
                                                         static_templates=static_templates,
                                                         single_particle_calibration=single_particle_calibration,
                                                         hard_baseline=hard_baseline,
                                                         particles_overlapping=particles_overlapping,
                                                         sweep_method=sweep_method,
                                                         sweep_param=sweep_param,
                                                         use_stack_id=calib_stack_id,
                                                         ).unpack()
                    elif test_dataset == '20X_1Xmag_5.1umNR_HighInt_0.03XHg':
                        test_settings = dataset_unpacker(dataset=test_dataset, collection_type='meta-test',
                                                         static_templates=False, single_particle_calibration=True, hard_baseline=False,
                                                         particles_overlapping=False, particle_distribution=particle_distribution,
                                                         sweep_method=sweep_method, sweep_param=sweep_param, use_stack_id=sweep_param).unpack()

                else:
                    raise ValueError("No dataset found.")
            else:
                # if test_dataset in ['wA2', 'wA_b1_test2', 'wA_a1_test1']:
                test_settings = dataset_unpacker(dataset=test_dataset,
                                                 collection_type='test',
                                                 static_templates=static_templates,
                                                 single_particle_calibration=single_particle_calibration,
                                                 hard_baseline=hard_baseline,
                                                 particles_overlapping=particles_overlapping,
                                                 particle_distribution=particle_distribution,
                                                 sweep_method=sweep_method,
                                                 sweep_param=sweep_param,
                                                 use_stack_id=None).unpack()
                #else:
                #    raise ValueError("No dataset found.")

            # ----- ----- ----- ----- TEST ----- ----- ----- ----- ----- ----- -----
            if calib_col is not None:
                GdpytCharacterize.test(calib_settings, test_settings, calib_col=calib_col, calib_set=calib_set,
                                       return_variables=None)
            else:
                GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None,
                                       return_variables=return_variables)

# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- ----- ----- ----

# ----- ----- ----- ----- Synthetic Experiment Validation - Static --- -- ----- ----- ----- ----- ----- ----- ----- ----
multi = False
if multi:

    calib_col, calib_set = None, None
    """
    test_dataset = '11.06.21_z-micrometer-v2'
    particle_distribution = 'SILPURAN'
    sweep_method = 'micrometer_5um'
    sweep_params = [[0, 5], [0, 3], [0, 0]]  #
    calib_stack_id = 46  # 'nearest'
    """
    single_particle_calibration = True
    static_templates = False
    hard_baseline = False
    particles_overlapping = False

    sweep_params = [41, 48, 52, 56]
    calib_stack_id = None  # 'nearest'

    # generate calibration collection and calibration set
    calib_settings = dataset_unpacker(dataset=test_dataset,
                                      collection_type='calibration',
                                      noise_level=nl,
                                      number_of_images=num_test_images,
                                      particle_distribution=particle_distribution,
                                      particle_density=None,
                                      single_particle_calibration=single_particle_calibration,
                                      static_templates=static_templates,
                                      hard_baseline=hard_baseline,
                                      particles_overlapping=particles_overlapping,
                                      sweep_method=sweep_method,
                                      sweep_param='spct-cal').unpack()

    calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')

    for sweep_param in sweep_params:
        """calib_settings = dataset_unpacker(dataset=test_dataset,
                                          collection_type='calibration',
                                          noise_level=nl,
                                          number_of_images=num_test_images,
                                          particle_distribution=particle_distribution,
                                          particle_density=sweep_param,
                                          single_particle_calibration=single_particle_calibration,
                                          static_templates=static_templates,
                                          hard_baseline=hard_baseline,
                                          particles_overlapping=particles_overlapping,
                                          sweep_method=sweep_method,
                                          sweep_param=sweep_param,
                                          use_stack_id=calib_stack_id).unpack()"""
        test_settings = dataset_unpacker(dataset=test_dataset,
                                         collection_type='test',
                                         noise_level=nl,
                                         number_of_images=num_test_images,
                                         particle_distribution=particle_distribution,
                                         particle_density=None,
                                         single_particle_calibration=single_particle_calibration,
                                         static_templates=static_templates,
                                         hard_baseline=hard_baseline,
                                         particles_overlapping=particles_overlapping,
                                         sweep_method=sweep_method,
                                         sweep_param=sweep_param,
                                         use_stack_id=sweep_param).unpack()

        if calib_col is not None:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=calib_col, calib_set=calib_set,
                                   return_variables=None)
        else:
            GdpytCharacterize.test(calib_settings, test_settings, calib_col=None, calib_set=None, return_variables=None)

# -----

print('Analysis completed without errors.')