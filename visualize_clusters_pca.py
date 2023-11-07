import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters_pca(features, kmeans):
    """
    Reduces the feature space to 2 dimensions using PCA and plots the clusters determined by the KMeans algorithm.

    Parameters:
    features (array-like): High-dimensional data to be visualized, where each row corresponds to an observation 
                           and each column corresponds to a feature.
    kmeans (KMeans or MiniBatchKMeans object): A fitted KMeans or MiniBatchKMeans clustering object from sklearn.

    Returns:
    None: The function directly shows the plot.

    The function assumes that the `kmeans` object has already been fitted to the data. It uses the cluster 
    assignments from the `kmeans.labels_` attribute to color the points in the plot.
    
    Examples:
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
    >>> kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    >>> visualize_clusters_pca(X, kmeans)
    """
    # Reduce the feature space to 2 dimensions using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Plot each cluster
    for i in range(kmeans.n_clusters):
        # Select only data observations with cluster label == i
        cluster_points = reduced_features[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    plt.legend()
    plt.title('Clusters visualization with PCA')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.show()
