import numpy as np
from hyperalg.distances import get_pair_distances as get_pair_distances_cpp
from hyperalg.distances import Distance, get_pair_distances_vec
from hyperalg.potentials.collective_coords import get_pair_distances

if __name__ == "__main__":
    ndim = 2
    natoms = 100
    npairs = int(natoms*(natoms-1)/2)
    bbox = np.ones(ndim)
    coords = np.random.rand(natoms, ndim)
    # coords = np.array([0, 0, 0.5, 0.5], dtype='d').reshape(-1, 2)
    pair_distances = np.zeros((npairs, 2), dtype='d')
    get_pair_distances(coords, pair_distances)
    pair_distances = np.linalg.norm(pair_distances, axis=1)
    print(pair_distances)

    pair_distances_cpp = get_pair_distances_cpp(coords.ravel(), ndim, Distance.PERIODIC, bbox)
    print(pair_distances_cpp)

    pair_distances_vec = get_pair_distances_vec(coords.ravel(), ndim, Distance.PERIODIC, bbox)
    pair_distances_vec = np.linalg.norm(pair_distances_vec.reshape(-1, 2), axis=1)
    print(pair_distances_vec)

    print(np.allclose(np.sort(pair_distances), np.sort(pair_distances_cpp)))
