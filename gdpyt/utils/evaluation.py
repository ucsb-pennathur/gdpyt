from gdpyt import GdpytImageCollection
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from os.path import join
from os import listdir
import logging

logger = logging.getLogger(__name__)

class GdpytPerformanceEvaluation(object):

    __name__ = 'GdpytPerformanceEvaluation'

    def __init__(self, image_collection, source_folder):
        assert isinstance(image_collection, GdpytImageCollection)
        if not image_collection.is_infered():
            logger.warning("{}: Some images in collection {} contain particles whose z coordinate "
                           "has not been infered".format(self.__name__, image_collection.folder))
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

            # Get coordinates of particles. Omit particles that have no z coordinate
            particle_locs = self._collection.images[file.split('.')[0] + img_ftype].particle_coordinates().dropna(subset=['z'])
            max_sims = self._collection.images[file.split('.')[0] + img_ftype].maximum_cm()
            max_sims = max_sims[max_sims['id'].isin(particle_locs['id'])]

            dist, idx = dist_tree.query(particle_locs[['x', 'y']].values, k=1, p=2)

            eval_df = pd.concat([particle_locs, pd.DataFrame({'x_true': source_data['x'].values[idx],
                                                              'y_true': source_data['y'].values[idx],
                                                              'z_true': source_data['z'].values[idx],
                                                              'delta_z': particle_locs['z'] - source_data['z'].values[idx],
                                                              'delta_xy': dist})], axis=1).join(max_sims.set_index('id'), on='id')
            perf_df.append(eval_df.set_index('id'))

        self.eval_df = pd.concat(perf_df, keys=[fname.split('.txt')[0] + img_ftype for fname in listdir(self._source_folder)], names=['Image', 'id'])

    def sigma_z(self, min_cm=None):
        """
        Silvan's implementation
        """
        if min_cm is not None:
            dz = self.eval_df.reset_index().query('Cm > {}'.format(min_cm))['delta_z'].values
        else:
            dz = self.eval_df.reset_index()['delta_z'].values
        return np.sqrt(np.power(dz, 2).sum() / len(dz))

    def sigma_z_local(self, bins=20, min_cm=None):
        """
        Silvan's implementation
        """
        assert bins > 1
        if min_cm is not None:
            z_df = self.eval_df.reset_index().query('Cm > {}'.format(min_cm))[['z_true', 'delta_z']]
        else:
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


def calculate_particle_to_particle_spacing(collection, max_dx=250, max_n_neighbors=5):
    """

    Returns
    -------

    """
    for img in collection.images.values():

        # GdpytParticles + locations
        particles = [particle for particle in img.particles]
        locations = np.array([list(p.location) for p in particles])

        nneigh = NearestNeighbors(n_neighbors=max_n_neighbors, algorithm='ball_tree').fit(locations)
        distances, indices = nneigh.kneighbors(locations)

        distance_to_others = distances[:, 1:]

        for distance, p in zip(distance_to_others, particles):

            frame = img.frame
            pid = p.id

            mean_diameter = np.mean([p.gauss_dia_x, p.gauss_dia_y])
            mean_radius = np.round(mean_diameter / 2, 2)

            mean_dx_all = np.mean(distance)
            min_dx_all = np.min(distance)
            num_dx_all = max_n_neighbors

            overlapping_dists = distance[distance < max_dx]
            mean_dxo = np.mean(overlapping_dists)
            num_dxo = len(overlapping_dists)

            pid_dx = [frame, pid, mean_radius, mean_dx_all, min_dx_all, num_dx_all, mean_dxo, num_dxo]