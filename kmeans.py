import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    # use generator whenever you want to use np.random

    centers = []
    center_index = generator.randint(0, n)
    centers.append(center_index)
    flag = 0
    for i in range(1, n_cluster):
        nearest_center = []
        for pt in range(len(x)):
            dist = []
            for c in range(len(centers)):
                if centers[c] == pt:
                    flag = 1
                    dist.append(1000000000)
                else:
                    dist.append(np.sum(np.square(x[centers[c]] - x[pt])))

            if flag == 1:
                nearest_center.append(-100000000)
            else:
                nearest_center.append(min(dist))
            flag = 0
        new_center = np.argmax(np.array(nearest_center))
        # center = x[new_center]
        centers.append(new_center)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))

    return centers


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DO NOT CHANGE CODE ABOVE THIS LINE

        centroids = np.array([x[i] for i in self.centers])  # center point, k X D
        iter = 0
        J = np.power(10, 10)

        while (iter != self.max_iter):

            # compute assignments: y
            y = []
            dist = np.reshape(np.sum(x ** 2, axis=1), (N, 1)) + np.sum(centroids ** 2, axis=1) - 2 * x.dot(
                centroids.T)  # dist:N x n_cluster
            y = np.argmin(dist, axis=1).tolist()

            # compute J_new
            J_new = np.sum([np.min(dist, axis=1)]) / N

            if np.abs(J - J_new) <= self.e:
                break
            else:
                J = J_new

            # compute centers:
            # build a dictionary with key:cluster_the pt belongs to , value: indices of pts that belongs to cluster[key]
            dict = {i: [] for i in range(self.n_cluster)}
            for i in range(N):
                dict[y[i]].append(i)

            centroids = centroids.tolist()
            for j in range(self.n_cluster):
                l = len(dict[j])
                if l == 0:
                    continue
                u = []
                for a in range(D):
                    u.append((np.sum([x[i][a] for i in dict[j]])) / l)
                centroids[j] = u

            centroids = np.array(centroids)

            iter += 1

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, np.array(y), iter


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''
        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # get centroids and membership
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, i = kmeans.fit(x, centroid_func)

        # compute centroids labels with majority votes from its membership
        # build a dictionary with key = cluster and value=label of point p
        dict = {i: [] for i in range(self.n_cluster)}
        for i in range(N):
            dict[membership[i]].append(y[i])
        centroid_labels = []
        for key, j in dict.items():
            counts = np.bincount(j, minlength=self.n_cluster)
            majority = np.argmax(counts)
            if len(j) == 0:
                centroid_labels.append(0)
            else:
                centroid_labels.append(majority)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = np.array(centroid_labels)
        self.centroids = np.array(centroids)

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - Implement the prediction algorithm
        # - return labels
        # DONOT CHANGE CODE ABOVE THIS LINE
        dist = np.reshape(np.sum(x ** 2, axis=1), (N, 1)) + np.sum(self.centroids ** 2, axis=1) - 2 * x.dot(
            self.centroids.T)  # dist:N x n_cluster
        center_index = [np.argmin(dist[j, :]) for j in range(N)]
        labels = [self.centroid_labels[i] for i in center_index]
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # care about the color(RGB)not distance
    x = image.copy()
    N = x.shape[0] * x.shape[1]
    D = x.shape[2]
    x = x.reshape(N, D)
    dist = np.reshape(np.sum(x ** 2, axis=1), (N, 1)) + np.sum(code_vectors ** 2, axis=1) - 2 * x.dot(
        code_vectors.T)  # dist:N x n_cluster
    y = np.argmin(dist, axis=1).tolist()

    for i in range(N):
        x[i] = code_vectors[y[i]]
    new_im = x.reshape(image.shape[0], image.shape[1], image.shape[2])

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im