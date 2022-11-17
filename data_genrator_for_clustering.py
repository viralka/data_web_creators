import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def generate_data(n_samples, n_features, n_clusters, n_outliers, n_inliers, n_noise, random_state):
    """
    Generate data for clustering
    :param n_samples: number of samples
    :param n_features: number of features
    :param n_clusters: number of clusters
    :param n_outliers: number of outliers
    :param n_inliers: number of inliers
    :param n_noise: number of noise
    :param random_state: random state
    :return: data
    """
    # generate data
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=random_state)
    # add outliers
    X = np.concatenate((X, np.random.uniform(low=-15, high=15, size=(n_outliers, n_features))), axis=0)
    # add inliers
    X = np.concatenate((X, np.random.uniform(low=-5, high=5, size=(n_inliers, n_features))), axis=0)
    # add noise
    X = np.concatenate((X, np.random.uniform(low=-15, high=15, size=(n_noise, n_features))), axis=0)
    # shuffle data
  
    return X, y

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, shuffle=True, random_state=None):
    """
    Generate isotropic Gaussian blobs for clustering.
    :param n_samples: The total number of points equally divided among clusters.
    :param n_features: The number of features for each sample.
    :param centers: The number of centers to generate, or the fixed center locations.
    :param cluster_std: The standard deviation of the clusters.
    :param shuffle: Shuffle the samples.
    :param random_state: Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.
    :return: X, y
    """
    # check parameters
    assert n_samples >= centers, "n_samples should be greater than or equal to n_centers"
    assert n_features > 0, "n_features should be greater than 0"
    assert cluster_std > 0, "cluster_std should be greater than 0"
    # generate data
    
    x = np.linspace(-10, 10, centers).reshape(centers, 1)
    np.random.shuffle(x)
    y = np.linspace(-10, 10, centers).reshape(centers, 1)
    np.random.shuffle(y)

    


    centers_vector = np.concatenate((x, y), axis=1)
    print(centers_vector)
    # centers_vector = np.random.normal( size=(centers, n_features))
    # centers_vector = np.cumsum(centers_vector, axis=0)
    # distance_between_centers = n_samples/centers * 2/
    # random_array = np.random.rand(centers)
    # print(random_array)

    # print(centers)

    

    return centers_vector



    # return X, y


for i in range(10):
    centers = make_blobs(n_samples=100, n_features=2, centers=10, cluster_std=1.0, shuffle=True, random_state=None)
    # print(centers)
    plt.scatter(centers[:,0], centers[:,1], c='red', s=50)
    
    plt.pause(2)

    plt.close()

    
plt.show()
    