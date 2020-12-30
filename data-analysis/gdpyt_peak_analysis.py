# NOTES ABOUT PROGRAM:
# --- This program calculates:
# ----- 1. particle deflection vs. time
# ----- 2. cross section deflection profile at peak deflection
# ----- 3. channel section deflection profile at peak deflection

#modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.draw import line, polygon, polygon_perimeter, circle, circle_perimeter
from skimage import (io)
import cv2


# ---------------------------------------------------------------------











"""
In the below section, we define scripts that are necessary for the data analysis:
1. test_img_plot --> opens, draws on, and save images.
2.
3.
4. 
5.
"""


# ---------------------------------------------------------------------


# 0 - DEFINE IMAGE LOAD, DRAW, SHOW, AND SAVE FUNCTION

def test_img_plot(img_name, img_read_loc, img_read_format='.tif', img_save_loc=None, img_save_format='png',
                  tly=123, bly=415, bry=427, trry=136,
                  part_cx=False, part_cy=False,
                  draw_bpe_color=None, draw_particle_color=None,
                  draw_particle=True, draw_bpe=True, show=True, save=False,
                  pause_time = 3, um_per_pix=1, bf=35, draw_particle_radius=10,
                  ):

    # other variables
    lr_shift_y = 14  # angle of image
    w = 550          # channel width

    # read file path
    img_name = img_name + '_X1'
    image_read = img_read_loc + img_name + img_read_format

    # load image
    img = io.imread(image_read)

    # calculate image shape
    img_dims = np.shape(img)

    # calculate BPE locations
    centerline = int(np.round((np.mean([tly, bly]) + np.mean([bry, trry])) / 2, 0))
    bpe_leadedge_centerline = int(np.round((np.mean([tly, bly]))))

    # draw bpe rectangle
    if draw_bpe == True:

        # choose color
        if draw_bpe_color == None:
            draw_bpe_color = int(np.round(bf * np.mean(img), 0))

        # organize particle data
        xloc = int(np.round((part_cx) / um_per_pix, 0))
        yloc = int(np.round((part_cy) / um_per_pix, 0))

        # draw rectangle
        poly = np.array(((yloc-thresh, xloc-thresh), (yloc-thresh, xloc+thresh),
                         (yloc+thresh, xloc+thresh), (yloc+thresh, xloc-thresh), (yloc-thresh, xloc-thresh)))

        rr, cc = polygon_perimeter(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc] = draw_bpe_color

        # rescale back to micron coordinates and adjust image name
        xloc = int(np.round(xloc * um_per_pix, 0))
        yloc = int(np.round(yloc * um_per_pix, 0))

        # draw centerline line
        #rr, cc = line(centerline, 0, centerline, 511)
        #img[rr, cc] = draw_bpe_color

        # draw wall lines
        rr, cc = line(int(np.round(centerline + (w / um_per_pix) / 2, 0) - 6), 0,
                      int(np.round(centerline + (w / um_per_pix) / 2 + lr_shift_y, 0) - 6), 511)
        img[rr, cc] = draw_bpe_color

        rr, cc = line(int(np.round(centerline - (w / um_per_pix) / 2, 0) - 6), 0,
                      int(np.round(centerline - (w / um_per_pix) / 2 + lr_shift_y, 0) - 6), 511)
        img[rr, cc] = draw_bpe_color


    # draw particle
    if draw_particle == True:

        # choose color
        if draw_particle_color == None:
            draw_particle_color = int(np.round(np.min(img), 0))

        # organize particle data
        xloc = int(np.round((part_cx) / um_per_pix, 0))
        yloc = int(np.round((part_cy) / um_per_pix, 0))

        # create circle perimeter
        rr, cc = circle_perimeter(yloc, xloc, draw_particle_radius)
        rr = np.where(rr>511, 511, rr)
        rr = np.where(rr<0, 0, rr)
        cc = np.where(cc>511, 511, cc)
        cc = np.where(cc < 0, 0, cc)
        img[rr, cc] = draw_particle_color

        # rescale back to micron coordinates
        xloc = int(np.round(xloc * um_per_pix, 0))
        yloc = int(np.round(yloc * um_per_pix, 0))

    # adjust image name
    img_name = "Image-Particle_X" + str(xloc) + '.Y' + str(yloc)

    # create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img * bf, cmap='gray',  extent=(0, img_dims[0]*um_per_pix, img_dims[1]*um_per_pix, 0))
    ax.set_title(img_name)
    plt.xlabel('Axial - Channel (um)')
    plt.ylabel('Transverse - Channel (um)')
    plt.grid(color='dimgray', alpha=0.25, linestyle='-', linewidth=0.25)


    # display
    if show == True:
        plt.show()
        plt.pause(pause_time)

    # save
    if save == True:
        image_save = img_save_loc + img_name + '.' + img_save_format
        #plt.imsave(image_save, img * bf, format=img_save_format, cmap='gray')
        #plt_img_save = img_save_loc + img_name + '_plt.' + img_save_format
        print(image_save)
        plt.savefig(image_save, format=img_save_format)

    plt.close(fig=None)

    # return centerline
    return centerline

# ----- END OF FUNCTION -----


# ---------------------------------------------------------------------


# DEFINE MERGE AND INTERPOLATE DATAFRAME

def df_merge_and_interpolate(df, pid, dframe, method='spline', order=2):

    # drop "t"
    df = df.drop(columns=['t'])

    # merge good data frames
    dfm = pd.merge_ordered(dframe, df, fill_method="fillna", on='Frame')

    # interpolate NaNs
    dfm1 = dfm.interpolate(method=method, order=order)
    #dfm2 = dfm1.interpolate(method='nearest')
    #dfi = dfm1.fillna(dfm1.mean(), inplace=True)
    dfi = dfm1.fillna(value=dfm1.mean(), limit=1)


    return (dfi)

# ----- END OF FUNCTION -----


# ---------------------------------------------------------------------



# 4.1 - DEFINE TIME LAGGED CROSS CORRELATION

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


# ----- END OF FUNCTION -----



# 4.4 - DEFINE SORTING SCRIPT

def myFunc(t):
    # load file
    filename = data_loc + t + '_' + data_set_type + '.csv'
    df = pd.read_csv(filename, header=0,
                     dtype={'Frame': int, 'id': int, 'x': float, 'y': float, 'z': float, 't': float})

    # determine unique particle ids
    return df.id.unique().size

# ----- END OF FUNCTION -----

# -----------------           ----------------            ---------------------





"""
The below script is what actually computes the 1D correlation and plots the data. The process flow is as follows:
1. Determine BPE and Channel coordinates by viewing example image.
2.
3.
4.

"""


# ----------------- LOAD ALL VARIABLES NECESSARY TO RUN SCRIPT  -----------------------


# 0.0 - PHYSICAL VARIABLES

# 0.1 - mechanics
wall_height = 10    # channel depth
# 0.2 - electronics
f_keithley = 0.1                   # (Hz) electric field square-wave switching frequency
displacement_period = 1/f_keithley  # the period over which particle displacement should occur
# 0.3 - optics
pix_per_100um = 61  # x-y scaling
um_per_pix = np.round(100 / 61, 2)  # microns per pixel
f = 3.4747          # (Hz) image acquisition rate (BX41)
f_old = 8.85        # (Hz) image acquisition rate accidentally used for data processing
frames = 220        # number of frames in experiment set

# ----- ----- -----

# 1.0 - DATA INPUT AND OUTPUT

# 1.1 - read initial image
img_read_loc = '/Users/mackenzie/Downloads/testing_09.18.20_analysis_10.08.20_BX41_IC10_870nmFP/tests/test_img_setup/'
img_name = 'test8_12Vmm'
img_read_format = '.tif'

# 1.2 - read initial data
data_loc = '/Users/mackenzie/Downloads/testing_09.18.20_analysis_10.08.20_BX41_IC10_870nmFP/results/results_10.13.20/'
data_set_type = 'coords_cnn_filtered'
tests = [
    'test0_0Vmm',
    'test2_1Vmm',
    'test3_2Vmm',
    'test4_4Vmm',
    'test5_6Vmm',
    'test6_8Vmm',
    'test7_10Vmm',
    'test8_12Vmm'
]

# 1.3 - data output locations (save)
img_save_loc ='/Users/mackenzie/Downloads/testing_09.18.20_analysis_10.08.20_BX41_IC10_870nmFP/results/results_11.3.20/'
img_save_format = 'png'

# ----- ----- -----

# 2.0 - IMAGE SHOW, DRAW, AND SAVE
bf = 35
draw_particle_color = 15000
pause_time = 5
draw_bpe_color = None
draw_particle = False
draw_bpe = True
show = False
save = True



# ----- ----- -----

# 3.0 - DATA ANALYSIS AND 1D CROSS CORRELATION

# 3.1 - data analysis
thresh = 55         # particle capture radius used for sorting general dataframes to particle specific dataframe
flip_z = 1         # invert z-direction
z_offset = 45      # offset to adjust z-position so z=0 is zero deflection plane (z_offset is subtracted from df.z)
frame_ratio = frames / 3      # filter low count particles

# 3.2 - 1D correlation
corr_shift = True
interp_method = 'spline'
corr_method = 'time lag cross corr'
order = 1

# ----- ----- -----

# 4.0 - PLOTTING TOOLS

# 4.1 - figure design
linewidth = 3
scattersize = 25
linestyle_raw = '-'
linestyle_interp = '-'
marker_raw = 'o'
marker_interp = 'o'








# ----------------- DETERMINE BPE COORDINATES -------------------------------------
"""
NOTES:
* this section is a bit of a legacy. I use this to plot and adjust the BPE rectangle on the initial image. Otherwise,
this script does not need to be included in the program. 
"""

# 1 - LOAD FIRST IMAGE AND DETERMINE CENTER LINE

centerline = test_img_plot(img_name,
                           img_read_loc=img_read_loc,
                           img_read_format='.tif',
                           img_save_loc=None,
                           img_save_format='png',
                           tly=123, bly=415, bry=427, trry=136,
                           part_cx=False, part_cy=False,
                           draw_bpe_color=None, draw_particle_color=None,
                           draw_particle=False, draw_bpe=True, show=False, save=False,
                           pause_time=pause_time, um_per_pix=um_per_pix, bf=bf, draw_particle_radius=thresh)

print("BPE, Y-centroid: " + str(np.round(centerline * um_per_pix, 1)) + " (um)")




# ------------------ RUN THE PROGRAM ----------------------------------------------



# 1 - CREATE DATA "FRAME"

convert_dict = {'Frame': int, 't': float}  # dictionary for column datatype
dframe = pd.DataFrame(data=np.column_stack((np.arange(frames+1, dtype=int), np.arange(frames+1, dtype=int) / f)),
                      columns=['Frame', 't'])
dframe = dframe.astype(convert_dict)  # chage datatypes
dframe = dframe.set_index('Frame')  # set index



# 2 - DETERMINE DATASET WITH MOST UNIQUE PARTICLES

tests_counts = tests.copy()
tests_counts.sort(key=myFunc, reverse=True)
filename = data_loc + tests_counts[0] + '_' + data_set_type + '.csv'
print(tests_counts)



# 3 - LOAD FIRST TEST DATASET

dff = pd.read_csv(filename, header=0, dtype={'Frame': int, 'id': int, 'x': float, 'y': float, 'z': float, 't': float})



# 4 - IDENTIFY AND LOCATE EACH PARTICLE

# copy reduced dataset
df_p = dff[["id", "x", "y"]]

# group by and take mean
mean_locs = df_p.groupby(['id'])['x', 'y'].mean().sort_values(by='id')

# number of identified particles
print("# Unique Particles in " + tests_counts[0] + ": " + str(mean_locs.index.size))



# 5 - FOR EACH PARTICLE IDENTIFIED IN mean_locs:

for p in range(mean_locs.index.size):

    # initialize loop
    i_corr = 0
    print("----- BEGINNING LOOP -----")
    print("Particle ID: " + str(p))



    # 5.1 - initialize figure

    fig = plt.figure(num=1, figsize=(10,7))

    # 5.2.2 - load data
    location = '/Users/mackenzie/Downloads/testing_09.18.20_analysis_10.08.20_BX41_IC10_870nmFP/results/results_11.3.20'\
               '/peak_analysis.v1/'
    filename = location + 'peak_analysis' + '.csv'
    df = pd.read_csv(filename, header=0, dtype={'E': int, 'x': float, 'y': float, 'f_e': float, 'p_index': float,
                                                'p_t': float, 'p_w': float, 'p_z': float, 'p_h': float})


    # 5.2.4 - filter particles by X and Y coordinates within threshold region:
    df_filtered = df[
        (df['x'] <= mean_locs.values[p][0] + thresh)               &
                    (df['x'] >= mean_locs.values[p][0] - thresh)   &
                    (df['y'] <= mean_locs.values[p][1] + thresh)   &
                    (df['y'] >= mean_locs.values[p][1] - thresh)
    ]
    print("mean locs: " + str((mean_locs.values[p][0])) + " with a thresh: " + str(thresh))

    # 5.2.4.5 - perform groupby and take mean of all values
    df_plot = df_filtered.groupby(['E']).mean()

    # other data
    df_std = df_filtered.groupby(['E']).std()

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.errorbar(df_plot.index, df_plot.p_h, yerr=df_std.p_h,
                fmt='o', ecolor='g', capthick=2, solid_capstyle='projecting', capsize=5)

    #ax_raw.plot(df_plot.t, df_plot.z, linestyle=linestyle_raw, linewidth=linewidth, label=lbl)
    plt.scatter(df_plot.index, df_plot.p_h, color='blue', s=55, marker=marker_raw, edgecolors='black', linewidth=3)


    plt.title('Peak Deflection: X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(np.round(mean_locs.values[p][1], 1)))
    plt.ylabel('Peak Deflection (um)')
    plt.xlabel('E (V/mm)')

    plt.xlim(0,15)
    plt.ylim(0,18)

    # 5.3 - add details to figure
    savename = 'X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(np.round(mean_locs.values[p][1], 1))
    savefile = img_save_loc + savename + '.' + img_save_format

    # 5.4 - save figure
    plt.savefig(savefile, format=img_save_format)
    plt.clf()


# --------------------------------------------------
    # plot peak z coordinate as a function of time

    # 5.1 - initialize figure

    # 5.2.4.5 - perform groupby and take mean of all values
    df_plot = df_filtered.groupby(['E']).mean()

    # other data
    df_std = df_filtered.groupby(['E']).std()

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.errorbar(df_plot.index, df_plot.p_w, yerr=df_std.p_w,
                 fmt='o', ecolor='g', capthick=2, solid_capstyle='projecting', capsize=5)

    # ax_raw.plot(df_plot.t, df_plot.z, linestyle=linestyle_raw, linewidth=linewidth, label=lbl)
    plt.scatter(df_plot.index, df_plot.p_w, color='blue', s=55, marker=marker_raw, edgecolors='black', linewidth=3)

    plt.title('Peak Width: X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(
        np.round(mean_locs.values[p][1], 1)))
    plt.ylabel('Peak Deflection (um)')
    plt.xlabel('E (V/mm)')

    #plt.xlim(0, 15)
    #plt.ylim(0, 18)

    # 5.3 - add details to figure
    savename = 'Peak Width: X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(np.round(mean_locs.values[p][1], 1))
    savefile = img_save_loc + savename + '.' + img_save_format

    # 5.4 - save figure
    plt.savefig(savefile, format=img_save_format)
    plt.clf()




    # --------------------------------------------------
    # plot peak z coordinate as a function of time

    # 5.1 - initialize figure

    # 5.2.4.5 - perform groupby and take mean of all values
    df_plot = df_filtered.groupby(['E']).mean()

    # other data
    df_std = df_filtered.groupby(['E']).std()

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.errorbar(df_plot.index, df_plot.p_z, yerr=df_std.p_z,
                 fmt='o', ecolor='g', capthick=2, solid_capstyle='projecting', capsize=5)

    # ax_raw.plot(df_plot.t, df_plot.z, linestyle=linestyle_raw, linewidth=linewidth, label=lbl)
    plt.scatter(df_plot.index, df_plot.p_z, color='blue', s=55, marker=marker_raw, edgecolors='black', linewidth=3)

    plt.title('Peak Z-Height: X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(
        np.round(mean_locs.values[p][1], 1)))
    plt.ylabel('Peak Deflection (um)')
    plt.xlabel('E (V/mm)')

    # plt.xlim(0, 15)
    # plt.ylim(0, 18)

    # 5.3 - add details to figure
    savename = 'Peak Z-Height: X' + str(np.round(mean_locs.values[p][0], 1)) + '_Y' + str(
        np.round(mean_locs.values[p][1], 1))
    savefile = img_save_loc + savename + '.' + img_save_format

    # 5.4 - save figure
    plt.savefig(savefile, format=img_save_format)
    plt.clf()















    # ----- EXIT PLOT TRAJECTORY LOOP -----

    # 5.2.6 - plot image with that particular particle outlined
    t='test7_10Vmm'
    test_img_plot(img_name=t,
                      img_read_loc=img_read_loc,
                      img_read_format=img_read_format,
                      img_save_loc=img_save_loc,
                      img_save_format=img_save_format,
                      tly=123, bly=415, bry=427, trry=136,
                      part_cx=mean_locs.values[p][0],
                      part_cy=mean_locs.values[p][1],
                      draw_bpe_color=draw_bpe_color,
                      draw_particle_color=draw_particle_color,
                      draw_particle=draw_particle,
                      draw_bpe=draw_bpe,
                      show=show,
                      save=save,
                      um_per_pix=um_per_pix,
                      bf=bf,
                      draw_particle_radius=thresh,
                      )





# ----- EXIT PARTICLE LOCATION LOOP -----


plt.close()




# ---------------------------------------------------------------------
