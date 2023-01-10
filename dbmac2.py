import numpy as np
import DBMAC2.ma as ma
import sklearn.cluster as cl
import shapely.geometry as spgeo
import util.datasets as datasets
import util.geometry as ugeo


def dbmac2(data, minr, maxr, step, sig):
    data_scaled = datasets.scale(data)
    cluster_set = []
    print("Scaled data: " + str(data_scaled))

    # MST.test
    p = datasets.mst_test(data_scaled)

    while p < sig:
        print("DBMAC2 iteration begins")
        # multiscale anaylsis to identify feature matrix
        feature_matrix, num_modes = ma.multiscale_analysis(data_scaled, minr, maxr, step, sig)
        # separate clusters from noise
        print("Feature matrix from MA: " + str(feature_matrix))
        print("Shape of feature matrix: " + str(feature_matrix.shape))
        res_kmeans = cl.KMeans(n_clusters=num_modes, n_jobs=-1).fit(feature_matrix)
        print("KMeans result: " + str(res_kmeans))
        print("KMeans labeling: " + str(res_kmeans.labels_))
        print("KMeans cluster centers: " + str(res_kmeans.cluster_centers_))

        cluster_center_max = res_kmeans.cluster_centers_[0][0]
        cluster_center_max_index = 0
        cluster_center_min = res_kmeans.cluster_centers_[0][0]
        cluster_center_min_index = 0

        for cluster_index in range(1, num_modes):
            if cluster_center_max < res_kmeans.cluster_centers_[cluster_index][0]:
                cluster_center_max = res_kmeans.cluster_centers_[cluster_index][0]
                cluster_center_max_index = cluster_index

            if cluster_center_min > res_kmeans.cluster_centers_[cluster_index][0]:
                cluster_center_min = res_kmeans.cluster_centers_[cluster_index][0]
                cluster_center_min_index = cluster_index

        print("Biggest cluster center: " + str(cluster_center_max) + " with index " + str(cluster_center_max_index))
        print("Smallest cluster center: " + str(cluster_center_min) + " with index " + str(cluster_center_min_index))
        cluster_points = []
        for i, val in enumerate(res_kmeans.labels_):
            if val == cluster_center_max_index:
                cluster_points.append(data_scaled[i])

        cluster_obj = np.asarray(cluster_points)
        print("Size of cluster object: " + str(len(cluster_obj)))
        print("Cluster object: " + str(cluster_obj))
        print("Shape of cluster object: " + str(cluster_obj.shape))
        # dbscan
        dbscan_res = cl.DBSCAN(eps=0.05, min_samples=3).fit(cluster_obj)
        print("DBSCAN result: " + str(dbscan_res.labels_))
        num_clusters = max(dbscan_res.labels_) + 1
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(dbscan_res.labels_):
            clusters[label].append(list(cluster_obj[i]))
        for cluster in clusters:
            cluster_set.append(np.asarray(cluster))

        """
        Calculate the density ratio as the quotient of cluster_center_max and cluster_center_min to get a measure of
        the relative density of the cluster, in order to be able to calculate back to the density of the noise
        """
        den_rat = cluster_center_max / cluster_center_min
        # Calculate the density of the cluster mode, according to DBSCAN result
        # First, calculate all the current cluster Multipoints (from shapely)
        clusters_mode_multipoints = list(spgeo.MultiPoint(datasets.list_of_lists_to_list_of_tuples(cluster))
                                         for cluster in clusters)
        cluster_polygons = list(multipoint.convex_hull for multipoint in clusters_mode_multipoints)
        # Calculate the sum of the areas of all clusters by using their convex hulls
        mode_area = sum(poly.area for poly in cluster_polygons)
        print("Area of current cluster mode: " + str(mode_area))
        # Calculate the density of the cluster mode and noise
        den_mode = len(cluster_obj) / mode_area
        den_noise = den_mode / den_rat
        print("Current cluster density ratio: " + str(den_rat))
        print("Current cluster mode density: " + str(den_mode))
        print("Current noise density: " + str(den_noise))

        """
        Remove the cluster and add back in a noise sample at the same area, based on the density of the actual noise
        """
        # Convert data and cluster object into sets of tuples to allow for set-theoretic operations
        data_scaled_as_set_of_tuples = set(tuple(point) for point in data_scaled)
        cluster_obj_as_set_of_tuples = set(tuple(point) for point in cluster_obj)
        # Subtract the cluster mode's clusters from the data
        data_scaled_as_set_of_tuples = data_scaled_as_set_of_tuples - cluster_obj_as_set_of_tuples
        # Calculate number of noise points needed to fill the hole
        num_needed_noise_points = int(mode_area * den_noise)
        # Add noise to the dataset where the clusters were
        data_scaled_as_set_of_tuples = data_scaled_as_set_of_tuples | \
                                       set(tuple(point) for point in ugeo.get_random_points_in_polygons(
                                           cluster_polygons, num_needed_noise_points))
        # Convert back the data to numpy array
        data_scaled = np.asarray(list(data_scaled_as_set_of_tuples))
        print("Data at the end of iteration: " + str(data_scaled))
        p = datasets.mst_test(data_scaled)

    return cluster_set, len(cluster_set)

