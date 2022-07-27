"""
This program tests the GDPyT measurement accuracy on... DataSet I
"""

from gdpyt import GdpytCharacterize
from gdpyt.utils.datasets import dataset_unpacker
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.exposure import rescale_intensity


# ----- ----- ----- ----- TEST DATASET UNPACKER ----- ----- ----- ----- ----- ----- -----

test_dataset = 'synthetic_overlap_noise-level'
particle_distribution = 'grid-dz-wide'
nl = 2
sweep_method = 'get'
sweep_params = ['idpt-figs']
calib_stack_id = 1

single_particle_calibration = False
static_templates = True
hard_baseline = True
particles_overlapping = True

# generate calibration collection and calibration set
calib_settings = dataset_unpacker(dataset=test_dataset,
                                  collection_type='calibration',
                                  particle_distribution=particle_distribution,
                                  noise_level=nl,
                                  single_particle_calibration=single_particle_calibration,
                                  static_templates=static_templates,
                                  hard_baseline=hard_baseline,
                                  particles_overlapping=particles_overlapping,
                                  sweep_method=sweep_method,
                                  sweep_param='gen_cal',
                                  use_stack_id=calib_stack_id).unpack()

calib_col, calib_set = GdpytCharacterize.test(calib_settings, test_settings=None, return_variables='calibration')


# --- FUNCTION CALL

# plot details
stepsize = 1
intensity_percentile = (20, 99)
aspect_ratio = 2.5

# id_ = 1
for id_ in calib_col.particle_ids:

    # ---- FUNCTION
    temp = []
    z = []
    for p in calib_set.calibration_stacks[id_].particles:
        temp.append(p.template)
        z.append(p.z_true)
    zipped = zip(z, temp)
    z_stack = list(sorted(zipped, key=lambda x : x[0]))

    stack_3d = []
    for p in z_stack:
        x, y = p[1].shape
        yh = int(y // 2)
        half_temp = p[1][:, :yh]
        stack_3d.append(half_temp)

    stack_3d = np.array(stack_3d)
    vmin, vmax = np.percentile(stack_3d, intensity_percentile)
    stack_rescale = rescale_intensity(stack_3d, in_range=(vmin, vmax), out_range=(0, 1))

    X, Y = np.mgrid[0:x, 0:yh]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect=(1, 1, aspect_ratio))

    # ls = LightSource(azdeg=40, altdeg=50) if you want to shade the image slices

    # get middle slice
    h, w, l = np.shape(stack_rescale)
    h_mid = int(np.floor(h / 2))

    for i, img in enumerate(stack_rescale):
        if i == h_mid:
            alpha = 1
        else:
            alpha = 0.25

        if i == 0 or (i + 1) % stepsize == 0:
            Z = np.zeros(X.shape) + z_stack[i][0]
            T = mpl.cm.RdBu(img)
            # T = ls.shade(img, plt.cm.viridis) if you want to shade the image slices
            fig3d = ax.plot_surface(X, Y, Z, facecolors=T, linewidth=0, alpha=alpha, cstride=1, rstride=1, antialiased=False)

    ax.view_init(40, 50)

    ax.set_title(r'Calibration stack: $p_{ID}$ = ' + str(id_))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$z_{true}$ / h')

    cbar = fig.colorbar(fig3d)
    cbar.ax.set_yticklabels(np.arange(vmin, vmax, np.floor((vmax-vmin)/5), dtype=int))
    cbar.set_label(r'$I_{norm}$')

    plt.show()
    j = 1