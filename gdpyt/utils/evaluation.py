from gdpyt import GdpytImageCollection
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
from os.path import join
from os import listdir


class GdpytPerformanceEvaluation(object):

    def __init__(self, image_collection, source_folder):
        assert isinstance(image_collection, GdpytImageCollection)
        if not image_collection.is_infered():
            raise RuntimeError("Can only evaluate performance on image collections that are fully inferred")
        self._collection = image_collection
        self._source_folder = source_folder
        self._evaluate()

    def _read_source(self, fname):
        if fname not in listdir(self._source_folder):
            raise FileNotFoundError("File {} not in {}".format(fname, self._source_folder))
        if '.txt' not in fname:
            raise ValueError("Filename must end with '.txt'. Received {}".format(fname))
        data = pd.read_csv(join(self._source_folder, fname), sep=' ', index_col=None, header=None)
        data.columns = ['x', 'y', 'z', 'd']

        return data

    def _evaluate(self):
        img_ftype = self._collection.filetype
        perf_df = []

        for file in listdir(self._source_folder):
            source_data = self._read_source(file)

            # Create KDTree for efficient distance lookup
            dist_tree = KDTree(source_data[['x', 'y']].values)
            particle_locs = self._collection.images[file.split('.')[0] + img_ftype].particle_coordinates()
            max_sims = self._collection.images[file.split('.')[0] + img_ftype].maximum_cm()

            dist, idx = dist_tree.query(particle_locs[['x', 'y']].values, k=1, p=2)

            eval_df = pd.concat([particle_locs, pd.DataFrame({'x_true': source_data['x'].values[idx],
                                                              'y_true': source_data['y'].values[idx],
                                                              'z_true': source_data['z'].values[idx],
                                                              'delta_z': particle_locs['z'] - source_data['z'].values[idx],
                                                              'delta_xy': dist})], axis=1).join(max_sims.set_index('id'), on='id')
            perf_df.append(eval_df.set_index('id'))

        self.eval_df = pd.concat(perf_df, keys=[fname.split('.txt')[0] + img_ftype for fname in listdir(self._source_folder)], names=['Image', 'id'])

    def sigma_z(self):
        dz = self.eval_df.reset_index()['delta_z'].values
        return np.sqrt(np.power(dz, 2).sum() / len(dz))

    def sigma_z_local(self, bins=20):
        assert bins > 1
        z_df = self.eval_df.reset_index()[['z_true', 'delta_z']]
        delta = (z_df['z_true'].max() - z_df['z_true'].min()) / (2*bins - 2)
        z_bins = np.linspace(z_df['z_true'].min() - delta, z_df['z_true'].max() + delta, bins + 1)
        sigma_df = pd.DataFrame()
        for i in range(bins):
            thisbin = z_df[(z_bins[i] < z_df['z_true']) & (z_df['z_true'] < z_bins[i + 1])]
            bin_center = (z_bins[i] + z_bins[i + 1]) / 2
            error_z = np.sqrt(np.power(thisbin['delta_z'].values, 2).sum() / len(thisbin))
            sigma_df = pd.concat([sigma_df, pd.DataFrame({'z': [bin_center], 'sigma_z_local': [error_z]})])
        return sigma_df

def error_z(image_collection, imgdata_folder=None):


    if imgdata_folder is None:
        raise NotImplementedError

    img_ftype = image_collection.filetype
    perf_df = []

    for file in listdir(imgdata_folder):
        assert '.txt' in file
        fname = file.split('.txt')[0]
        img_data = pd.read_csv(join(imgdata_folder, file), sep=' ', index_col=None, header=None)
        img_data.columns = ['x', 'y', 'z', 'd']

        # Create KDTree for efficient distance lookup
        dist_tree = KDTree(img_data[['x', 'y']].values)
        particle_locs = image_collection.images[fname + img_ftype].particle_coordinates()

        dist, idx = dist_tree.query(particle_locs[['x', 'y']].values, k=1, p=2)

        eval_df = pd.concat([particle_locs, pd.DataFrame({'actual_z': img_data['z'].values[idx],
                                                          'delta_z': particle_locs['z']-img_data['z'].values[idx],
                                                          'delta_xy': dist})], axis=1)
        perf_df.append(eval_df.set_index('id'))

    perf_df = pd.concat(perf_df, keys=[fname.split('.txt')[0] + img_ftype for fname in listdir(imgdata_folder)])

    return perf_df





        

