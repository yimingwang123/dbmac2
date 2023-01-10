import numpy as np
import unidip.unidip
import unidip.dip as dip
import scipy.spatial as sp


def multiscale_analysis(data: np.ndarray, minr: float, maxr: float, step: float, sig=0.05, unidip_alpha=0.05,
                        unidip_ntrials=100, unidip_mrg_dst=20):
    """

    :param data: The data points as a numpy array of shape (num_points, 2)
    :param minr: The minumum circle radius used for neighbor search
    :param maxr: The maximum circle radius
    :param step: The radius step
    :param sig: If the p-value is smaller than sig, the data will be considered multimodal - hence a small value
    :param unidip_alpha:
    :param unidip_ntrials:
    :param unidip_mrg_dst:
    :return: A tuple (feature_matrix, num_modes), where feature_matrix is a numpy array of shape (num_points, num_radi)
    where num_points is the number of points in data and num_radi is the number of radi that ma will calculate a ball
    for. num_modes is the number of modes that ma found in data
    """
    r = minr
    # We calculate the algorithm with a transposed feature matrix since it is easier
    feature_matrix_transposed = []
    num_modes = 0
    # neighborhood statistics
    print("MA start")
    while r < maxr:
        print("Calculating for r = " + str(r))
        # query_ball returns a struct that contains the neighbor points, which get counted by the lambda function
        tree = sp.KDTree(data)
        neighbor_struct = tree.query_ball_tree(tree, r)
        # -1 means to minus the point itself from all its neighbors
        # feature_matrix_row = list(map(lambda a: len(a) - 1, neighbor_struct))
        feature_matrix_row = list(len(neighbors) - 1 for neighbors in neighbor_struct)
        print("Resulting feature matrix row: " + str(feature_matrix_row))

        # create the discriminatory feature matrix
        feature_matrix_row_sorted = np.sort(np.copy(feature_matrix_row))  # ascending sort of the feature matrix row
        feature_matrix_row_sorted_noise = feature_matrix_row_sorted + np.random.normal(0, 0.01,
                                                                                       len(feature_matrix_row_sorted))
        (_, p, _) = dip.diptst(feature_matrix_row_sorted_noise)  # run hartigan's dip test to get the p-value
        if p is None:
            p = 1.0
        print('p-value: ' + str(p))
        if p < sig:
            print("p < sig!")
            n = len(unidip.UniDip(feature_matrix_row_sorted_noise, alpha=unidip_alpha, ntrials=unidip_ntrials,
                                  mrg_dst=unidip_mrg_dst).run())
            print('n = ' + str(n))
            if not feature_matrix_transposed:  # feature_matrix is empty
                print("Appending first row")
                feature_matrix_transposed.append(feature_matrix_row)
                num_modes = n
            else:
                if n == num_modes:
                    print("Appending row")
                    feature_matrix_transposed.append(feature_matrix_row)
                else:
                    break
        elif feature_matrix_transposed:
            break  # break if feature_matrix is not empty
        r += step

    print("MA end")
    return np.transpose(feature_matrix_transposed), num_modes
