# dbmac2

Finding meaningful clustering patterns in data can be very challenging when the clusters are of arbitrary shapes, different sizes, or densities, and especially when the data set contains high percentage (e.g., 80%) of noise. Unfortunately, most existing clustering techniques, such as kmeans, and Density-Based Multiscale Analysis for Clustering (or DBMAC, https://link.springer.com/chapter/10.1007/978-3-642-82937-6_5) cannot properly handle this tough situation and often result in dramatically deteriorating performance. Therefore, a purposefully designed clustering algorithm called Density-Based Multiscale Analysis for Clustering (DBMAC)-II is proposed, which is an improved version of the latest strong-noise clustering algorithm DBMAC. 

The pseudo-algo is elaborated in this paper: https://ieeexplore.ieee.org/abstract/document/8359265. 

### **Important Note:**
1. Both DBMAC and DBMAC2 are implemented in **python** in this repo.
2. The multiscale analysis is a supportive function for DBMAC2.
