
import numpy as np

class KMeansScratch:
    def __init__(self, k, max_iters=200, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_centroids = self._update(X, labels)

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def _assign(self, X):
        d = np.linalg.norm(X[:,None] - self.centroids, axis=2)
        return np.argmin(d, axis=1)

    def _update(self, X, labels):
        new=[]
        for i in range(self.k):
            pts=X[labels==i]
            if len(pts)==0:
                new.append(self.centroids[i])
            else:
                new.append(pts.mean(axis=0))
        return np.array(new)

    def inertia(self, X):
        return np.sum((X - self.centroids[self.labels_])**2)
