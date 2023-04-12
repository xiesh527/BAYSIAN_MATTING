import numpy as np


# Weighted mean
def calculate_weighted_mean(X, wg):
    Weighted_mean = np.sum(wg)
    weighted_X = np.multiply(X, wg[:, np.newaxis])
    mean = np.sum(weighted_X, axis=0) / Weighted_mean
    return mean

# Weighted covariance
def calculate_weighted_covariance(X, wg, mean):
    Weighted_mean = np.sum(wg)
    diff = X - np.broadcast_to(mean, X.shape)
    z_weighted = np.multiply(diff, np.sqrt(wg)[:, np.newaxis])
    covar = np.dot(z_weighted.T, z_weighted) / Weighted_mean + 1e-5 * np.eye(3)
    return covar

# Creating a node
def get_node(X, wg):

    # Calculating the mean, covariance, and number of pixels in the node
    mean = calculate_weighted_mean(X, wg)
    covar = calculate_weighted_covariance(X, wg, mean)
    N = X.shape[0]

    # Calculating the eigenvalues and eigenvectors
    V, D = np.linalg.eig(covar)
    lmbda = np.max(np.abs(V))
    e = D[np.argmax(np.abs(V))]

    # Creating the node using dictionary and returning it
    node = {'X': X, 'wg': wg, 'mean': mean, 'covar': covar, 'N': N, 'lmbda': lmbda, 'e': e, }
    return node

# Clustering the image
def clustOBouman(pix_val, pix_wg, minVar=0.05):

    # Creating a list of nodes
    nodes = []
    nodes.append(get_node(pix_val, pix_wg))

    # Branching the nodes until the variance is above the minimum variance
    while max(nodes, key=lambda x: x['lmbda'])['lmbda'] > minVar:

        # Branching the node with the largest eigenvalue
        idx_max = max(enumerate(nodes), key=lambda x: x[1]['lmbda'])[0]
        Clust_i = nodes[idx_max]

        # Calculating the splitting line
        idx = np.dot(Clust_i['X'], Clust_i['e']) <= np.dot(Clust_i['mean'], Clust_i['e'])

        # Creating the two children
        Clust_child1 = get_node(Clust_i['X'][idx], Clust_i['wg'][idx])
        Clust_child2 = get_node(Clust_i['X'][np.logical_not(idx)], Clust_i['wg'][np.logical_not(idx)])

        # Popping the parent and adding the child clusters
        nodes.pop(idx_max)
        nodes.append(Clust_child1)
        nodes.append(Clust_child2)

    # Extracting the mean and covariance from the nodes
    mean = np.array([node['mean'] for node in nodes])
    covar = np.array([node['covar'] for node in nodes])

    return mean, covar
