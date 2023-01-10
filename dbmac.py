from typing import List

import numpy as np
import DBMAC2.ma as ma
import sklearn.cluster as cl
import util.datasets as datasets

def dbmac(data, minr, maxr, step, sig):
    data_scaled = datasets.scale(data)

    print("Scaled data")
    print(str(data_scaled))
    # multiscale anaylsis to identify feature matrix
    feature_matrix, num_modes = ma.multiscale_analysis(data_scaled, minr, maxr, step, sig)
    # separate clusters from noise
    # 2 = k, 0 = random_state
    print("Feature matrix from MA: " + str(feature_matrix))
    print("Shape of feature matrix: " + str(feature_matrix.shape))
    res_kmeans = cl.KMeans(n_clusters=2, n_jobs=-1).fit(feature_matrix)
    print("KMeans result: " + str(res_kmeans))
    print("KMeans labeling: " + str(res_kmeans.labels_))
    print("KMeans cluster centers: " + str(res_kmeans.cluster_centers_))
    if res_kmeans.cluster_centers_[0][0] < res_kmeans.cluster_centers_[1][0]:
        cluster_center_max = res_kmeans.cluster_centers_[1][0]
        cluster_center_max_index = 1
    else:
        cluster_center_max = res_kmeans.cluster_centers_[0][0]
        cluster_center_max_index = 0
    print("Bigger cluster center: " + str(cluster_center_max) + " with index " + str(cluster_center_max_index))
    cluster_points: List = []
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
    return list(map(np.asarray, clusters)), num_clusters


