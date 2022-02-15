import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, draw
from skimage.draw import circle_perimeter_aa



# setup
path_name = '/Users/mackenzie/Desktop/test_coords_pd10.xlsx' # '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/synthetic random density uniform z nl1/test_coords/test_id2_coords_static_random_pd2.5e-3_uniform-z-nl1.xlsx'

# read dataframe
df = pd.read_excel(io=path_name, dtype=float)

"""
# get ids of all particles where c_m > 0.85 and again at 0.86
cm_low = 0.82
idls = df.loc[df['cm'] < cm_low].id.unique()
print(idls)

min_true_z = df.z_true.min()
dfz = df.loc[df['z_true'] == min_true_z]

xbool = dfz.id.isin(idls)


dfxy = dfz[xbool]

dfxy['yy'] = 1024 - dfxy['y']

xs = dfxy.x.to_numpy()
ys = dfxy.y.to_numpy()

print(xs)
print(ys)


img_path = '/Users/mackenzie/Desktop/calib_-15.07538.tif'
img = io.imread(img_path, plugin='tifffile')

imgg = img # np.zeros_like(img, dtype=np.uint16)

for x, y in zip(xs, ys):
    x = int(np.round(x, 0))
    y = int(np.round(y, 0))

    rr, cc, val = circle_perimeter_aa(x, y, 20)
    imgg[rr, cc] = 1 * 2 ** 16
    draw.set_color(imgg, (rr, cc), [1], alpha=1)

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(xs, ys, color='red', alpha=0.25)
plt.show()"""

# plot number of particles by cm

df.loc[:, 'cm'] = np.round(df.loc[:, 'cm'], 2)
dfg = df.groupby(by='cm').count()
fig, ax = plt.subplots()
ax.semilogy(dfg.index, dfg.z)
ax.set_xlabel(r'$c_m$')
ax.set_ylabel(r'$\#\quad particles$')
plt.show()

j=1