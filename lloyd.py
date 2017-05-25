import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def lloyd(images, k):
    clusters = np.array_split(images, k)
    centers = [0]*k
    for i in range(k):
        centers[i] = np.mean(clusters[i], axis=0)
    changed = True
    while changed:
        changed = False
        temp = [[] for i in range(k)]
        for image in images:
            dist = np.sum(np.subtract(centers, image)**2, axis=1)
            index = np.argmin(dist)
            temp[index].append(image)
        clusters = temp
        for i in range(k):
            temp = np.mean(clusters[i], axis=0)
            if not np.array_equal(temp, centers[i]):
                changed = True
            centers[i] = temp
    return centers, clusters

data = loadmat('mnist_data/images.mat', mat_dtype=True)
data = data['images']
data = np.reshape(data, (-1, 60000)).T
data = shuffle(data)
for k in (5, 10, 20):
    centers, clusters = lloyd(data, k)
    for i in range(k):
        center = np.reshape(centers[i], (28, 28))
        plt.imshow(center)
        plt.savefig('{}-{}.png'.format(k, i+1))
        plt.show()
