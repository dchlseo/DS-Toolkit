import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def find_optimal_clusters(data, max_k):
    """
    This function calculates and plots the Sum of Squared Errors (SSE) for different numbers of clusters 
    using the MiniBatchKMeans clustering algorithm. It is used to determine the optimal number of clusters 
    by observing the 'elbow' in the plot of SSE values.

    Parameters:
    data (array-like or DataFrame): The data to cluster. It should be in a format suitable for 
                                    MiniBatchKMeans (e.g., NumPy array, Pandas DataFrame).
    max_k (int): The maximum number of clusters to try. The function will test all even numbers of clusters
                 from 2 up to and including max_k.

    Returns:
    matplotlib.figure.Figure: A matplotlib Figure object with a plot of the SSE for each number of clusters.

    The 'elbow' method is visual and subjective; it looks for a cluster count after which the rate of decrease 
    of SSE sharply changes, indicating diminishing returns by increasing the number of clusters.
    
    Note:
    - It only considers even numbers of clusters within the specified range.
    - MiniBatchKMeans is used instead of KMeans for efficiency on large datasets. It uses a random sample
      of the data for each mini-batch to speed up the clustering process.

    Examples:
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
    >>> fig = find_optimal_clusters(X, max_k=10)
    >>> fig.show()
    """
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=42).fit(data).inertia_)
        print(f'Fit {k} clusters')
        
    # Plotting the inertia to see which number of clusters is best
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Number of Cluster Centers')
    
    return f

#################
## How to use this function
'''
# Run the function to find the optimal clusters for the sample data
optimal_clusters_figure = find_optimal_clusters(tfidf_matrix, 50)

# Display the plot
plt.show()
'''
