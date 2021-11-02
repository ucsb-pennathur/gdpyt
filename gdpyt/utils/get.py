# get.py
"""
Methods to get specific collections, images, or values from a larger collection.
"""
import numpy as np


def particles_with_large_z_error(error, test_col, particle_id=None, image_id=None, return_val=None):
    """
    Get particles with large z-error in a test collection.
    """
    particles = []

    if particle_id is not None:
        for image in test_col.images.values():
            for particle in image.particles:
                if particle.id == particle_id:
                    if np.isnan(particle.z):
                        particles.append(particle)
                    elif np.abs(particle.z_true - particle.z) > error:
                        particles.append(particle)
    elif image_id is not None:
        for particle in test_col.images[image_id]:
            if np.isnan(particle.z):
                particles.append(particle)
            elif np.abs(particle.z_true - particle.z) > error:
                particles.append(particle)
    else:
        for image in test_col.images.values():
            for particle in image.particles:
                if np.isnan(particle.z):
                    particles.append(particle)
                elif np.abs(particle.z_true - particle.z) > error:
                    particles.append(particle)

    if return_val == 'id':
        return [x.id for x in particles]
    else:
        return particles


def corresponding_calibration_stack_particle_template(calib_set, z, particle_id):
    """
    Get calibration image template nearest to z-coordinate.
    """

    # get the corresponding calibration stack to particle_id:
    if particle_id in calib_set.calibration_stacks.keys():
        stack = calib_set.calibration_stacks[particle_id]
    else:
        stack = calib_set.calibration_stacks[0]

    # get the z-coordinate and image template nearest to the z-coordinate
    z_calibs = list(stack.layers.keys())

    # get the index and z-difference closest to the z-coordinate
    index, z_diff = min(enumerate(z_calibs), key=lambda x: abs(x[1] - z))

    # get the z-coordinate and image template nearest to the z-coordinate
    z_calib = z_calibs[index]
    template_calib = stack.layers[z_calib]

    return z_calib, template_calib