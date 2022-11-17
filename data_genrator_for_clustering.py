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
    X, y = shuffle(X, y, random_state=random_state)
    # return data
    return X, y

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """
    Generate isotropic Gaussian blobs for clustering.
    :param n_samples: The total number of points equally divided among clusters.
    :param n_features: The number of features for each sample.
    :param centers: The number of centers to generate, or the fixed center locations.
    :param cluster_std: The standard deviation of the clusters.
    :param center_box: The bounding box for each cluster center when centers are generated at random.
    :param shuffle: Shuffle the samples.
    :param random_state: Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.
    :return: X, y
    """
    # check parameters
    assert n_samples >= centers, "n_samples should be greater than or equal to n_centers"
    assert n_features > 0, "n_features should be greater than 0"
    assert cluster_std > 0, "cluster_std should be greater than 0"
    assert center_box[0] < center_box[1], "center_box should be (min, max)"
    # generate data
    generator = check_random_state(random_state)
    centers = generator.uniform(center_box[0], center_box[1], size=(centers, n_features))
    X = []
    y = []
    for i in range(centers.shape[0]):
        X.append(centers[i] + generator.normal(scale=cluster_std, size=(int(n_samples / centers.shape[0]), n_features)))
        y.append(np.full(int(n_samples / centers.shape[0]), fill_value=i, dtype=np.int))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # shuffle data
    if shuffle:
        X, y = shuffle(X, y, random_state=random_state)
    # return data
    return X, y

def shuffle(X, y, random_state=42):                 #! 
    """
    Shuffle data
    :param X: data
    :param y: labels
    :param random_state: random state
    :return: X, y
    """
    # shuffle data
    generator = check_random_state(random_state)
    indices = np.arange(X.shape[0])
    generator.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # return data
    return X, y

def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance
    :param seed: None
    :return: np.random.RandomState instance
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    # if isinstance(seed, (numbers.Integral, np.integer)):
    #     return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


if __name__ == "__main__":
    # generate data
    
    random_state = 42

    X, y = generate_data(n_samples=1000, n_features=2, n_clusters=3, n_outliers=100, n_inliers=100, n_noise=100, random_state=0)
    # plot data
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()