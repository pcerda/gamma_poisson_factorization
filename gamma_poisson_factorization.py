import warnings
import numpy as np

from scipy.special import kl_div
from scipy import sparse

from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils import check_array
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster.k_means_ import _k_init


class OnlineGammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Online Non-negative Matrix Factorization by minimizing the
    Kullback-Leibler divergence.

    Parameters
    ----------

    n_topics: int, default=10
        Number of topics of the matrix Factorization.

    batch_size: int, default=100

    gamma_shape_prior: float, default=1.1
        Shape parameter for the Gamma prior distribution.

    gamma_scale_prior: float, default=1.0
        Shape parameter for the Gamma prior distribution.

    r: float, default=1
        Weight parameter for the update of the W matrix

    hashing: boolean, default=False
        If true, HashingVectorizer is used instead of CountVectorizer.

    hashing_n_features: int, default=2**10
        Number of features for the HashingVectorizer. Only relevant if
        hashing=True.

    tol: float, default=1E-3
        Tolerance for the convergence of the matrix W

    mix_iter: int, default=2

    max_iter: int, default=10

    ngram_range: tuple, default=(2, 4)

    init: str, default 'k-means++'
        Initialization method of the W matrix.

    random_state: default=None

    Attributes
    ----------

    References
    ----------
    """

    def __init__(self, n_topics=10, batch_size=512, gamma_shape_prior=1.1,
                 gamma_scale_prior=1.0, r=.7, hashing=False,
                 hashing_n_features=2**12, init='k-means',
                 tol=1E-4, min_iter=2, max_iter=5, ngram_range=(2, 4),
                 add_words=False, random_state=None, fisher_kernel=False):

        self.ngram_range = ngram_range
        self.n_topics = n_topics
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.r = r
        self.batch_size = batch_size
        self.tol = tol
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.add_words = add_words
        self.random_state = check_random_state(random_state)
        self.fisher_kernel = fisher_kernel

        if self.hashing:
            self.ngrams_count = HashingVectorizer(
                 analyzer='char', ngram_range=self.ngram_range,
                 n_features=self.hashing_n_features,
                 norm=None, alternate_sign=False)
            if self.add_words:
                self.word_count = HashingVectorizer(
                     analyzer='word',
                     n_features=self.hashing_n_features,
                     norm=None, alternate_sign=False)
        else:
            self.ngrams_count = CountVectorizer(
                 analyzer='char', ngram_range=self.ngram_range)
            if self.add_words:
                self.word_count = CountVectorizer()

    def _rescale_W(self, W, A, B):
        s = W.sum(axis=1, keepdims=True)
        W /= s
        A /= s
        return W, A, B

    def _rescale_H(self, V, H):
        epsilon = 1e-10  # in case of a document having length=0
        H *= np.maximum(epsilon, V.sum(axis=1).A)
        H /= H.sum(axis=1, keepdims=True)
        return H

    def _e_step(self, Vt, W, Ht, max_iter=20, epsilon=1E-3):
        WT1 = np.sum(W, axis=1) + 1 / self.gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
        const = (self.gamma_shape_prior - 1) / WT1
        squared_epsilon = epsilon**2
        for vt, ht in zip(Vt, Ht):
            vt_ = vt.data
            idx = vt.indices
            W_WT1_ = W_WT1[:, idx]
            squared_norm = 1
            for iter in range(max_iter):
                if squared_norm <= squared_epsilon:
                    break
                htW = np.dot(ht, W_WT1_)
                aux = np.dot(W_WT1_, vt_ / htW)
                ht_out = ht * aux + const
                squared_norm = np.dot(
                    ht_out - ht, ht_out - ht) / np.dot(ht, ht)
                ht[:] = ht_out
        return Ht

    def _m_step(self, Vt, W, A, B, Ht):
        A *= self.rho
        A += W * (
            Vt.multiply(np.dot(Ht, W)**-1).transpose().dot(Ht)).transpose()
        B *= self.rho
        B += Ht.sum(axis=0).reshape(-1, 1)
        return self._rescale_W(A / B, A, B)

    def _init_vars(self, X):
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.fit_transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        if not self.hashing:
            self.vocabulary = self.ngrams_count.get_feature_names()
            if self.add_words:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count.get_feature_names()))

        _, self.n_vocab = unq_V.shape
        self.W, self.A, self.B = self._init_W(unq_V[lookup], X)
        unq_H = self._rescale_H(unq_V, np.ones((len(unq_X), self.n_topics)))
        self.H_dict = dict()
        self._update_H_dict(unq_X, unq_H)
        self.rho = self.r**(self.batch_size / len(X))
        return unq_X, unq_V, lookup

    def _get_H(self, X):
        H_out = np.empty((len(X), self.n_topics))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict[x]
        return H_out

    def _init_W(self, V, X):
        if self.init == 'k-means++':
            W = _k_init(
                V, self.n_topics, row_norms(V, squared=True),
                random_state=self.random_state,
                n_local_trials=None) + .1
        elif self.init == 'random':
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
                size=(self.n_topics, self.n_vocab))
        elif self.init == 'k-means':
            prototypes = get_kmeans_prototypes(
                X, self.n_topics, random_state=self.random_state)
            W = self.ngrams_count.transform(prototypes).A + .1
            if self.add_words:
                W2 = self.word_count.transform(prototypes).A + .1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_topics:
                W2 = _k_init(
                    V, self.n_topics - W.shape[0], row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None) + .1
                W = np.concatenate((W, W2), axis=0)
        else:
            raise AttributeError(
                'Initialization method %s does not exist.' % self.init)
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_topics, self.n_vocab)) * 1E-10
        B = A.copy()
        return W, A, B

    def _update_H_dict(self, X, H):
        for x, h in zip(X, H):
            self.H_dict[x] = h

    def fit(self, X, y=None, X_test=None):
        """Fit the OnlineGammaPoissonFactorization to X.

        Parameters
        ----------
        X : string array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature
        Returns
        -------
        self
        """
        assert X.ndim == 1
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        unq_H = self._get_H(unq_X)

        for iter in range(self.max_iter):
            for i, (unq_idx, idx) in enumerate(batch_lookup(
              lookup, n=self.batch_size)):
                if i == n_batch-1:
                    W_last = self.W
                unq_H[unq_idx] = self._e_step(
                    unq_V[unq_idx], self.W, unq_H[unq_idx])
                self.W, self.A, self.B = self._m_step(
                    unq_V[idx], self.W, self.A, self.B, unq_H[idx])

                if i == n_batch-1:
                    W_change = np.linalg.norm(
                        self.W - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (iter >= self.min_iter - 1):
                break

        self._update_H_dict(unq_X, unq_H)
        return self

    def score(self, X):
        '''
        Returns the Kullback-Leibler divergence.

        Parameters
        ----------
        X : array-like (str), shape [n_samples,]
            The data to encode.

        Returns
        -------
        kl_divergence : float.
            Transformed input.
        '''

        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for Ht, Vt in batch2(unq_H, unq_V, n=self.batch_size):
            Ht[:] = self._e_step(Vt, self.W, Ht, max_iter=100)
        kl_divergence = kl_div(
            unq_V[lookup].A, np.dot(unq_H[lookup], self.W)).sum() / len(X)
        return kl_divergence

    def partial_fit(self, X, y=None):
        assert X.ndim == 1
        if hasattr(self, 'vocabulary'):
            unq_X, lookup = np.unique(X, return_inverse=True)
            unq_V = self.ngrams_count.transform(unq_X)
            if self.add_words:
                unq_V2 = self.word_count.transform(unq_X)
                unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

            unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict]))
            unseen_V = self.ngrams_count.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

            if unseen_V.shape[0] != 0:
                unseen_H = self._rescale_H(
                    unseen_V, np.ones(len(unseen_X), self.n_topics))
                for x, h in zip(unseen_X, unseen_H):
                    self.H_dict[x] = h
                del unseen_H
            del unseen_X, unseen_V
        else:
            unq_X, unq_V, lookup = self._init_vals(X)
            self.rho = .9  # fixed arbitrary value if no previous fit.

        unq_H = self._get_H(unq_X)
        unq_H = self._e_step(unq_V, self.W, unq_H)
        self._update_H_dict(unq_X, unq_H)
        self.W, self.A, self.B = self._m_step(
            unq_V[lookup], self.W, self.A, self.B, unq_H[lookup])
        return self

    def _add_unseen_keys_to_H_dict(self, X):
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

            unseen_H = self._rescale_H(
                unseen_V, np.ones((unseen_V.shape[0], self.n_topics)))
            self._update_H_dict(unseen_X, unseen_H)

    def transform(self, X):
        """Transform X using the trained matrix W.

        Parameters
        ----------
        X : array-like (str), shape [n_samples,]
            The data to encode.

        Returns
        -------
        X_new : 2-d array, shape [n_samples, n_topics]
            Transformed input.
        """
        unq_X = np.unique(X)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for Ht, Vt in batch2(unq_H, unq_V, n=self.batch_size):
            Ht[:] = self._e_step(Vt, self.W, Ht, max_iter=100)
        self._update_H_dict(unq_X, unq_H)

        if self.fisher_kernel:
            # TODO: improve memory usage
            H = self._get_H(X)
            V = self.ngrams_count.transform(X)
            if self.add_words:
                V2 = self.word_count.transform(X)
                V = sparse.hstack((V, V2), format='csr')
            fisher_score = V.multiply(np.dot(H, self.W)**-1)
            return fisher_score
        else:
            return self._get_H(X)


class GammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Gamma-Poisson factorization model (Canny 2004)
    """

    def __init__(self, n_topics=10, max_iters=100, fisher_kernel=False,
                 gamma_shape_prior=1.1, gamma_scale_prior=1.0, tol=.001,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_topics = n_topics  # parameter k
        self.max_iters = max_iters
        self.fisher_kernel = fisher_kernel
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.tol = tol

    def _mean_change(self, X_, X_last):
        scaled_diff = np.array(abs(X_ - X_last)) / np.array(X_last.sum(axis=0))
        mean_change = scaled_diff.sum(axis=0).mean()
        return mean_change

    def _rescale_Lambda(self, Lambda):
        factors = 1 / Lambda.sum(axis=0)
        Lambda_out = factors.reshape(1, -1) * Lambda
        return Lambda_out

    def _rescale_X(self, F, X):
        epsilon = 1e-10  # in case of a document having length=0
        doc_length = np.maximum(epsilon, np.array(F.sum(axis=0))).reshape(-1)
        X_length = X.sum(axis=0)
        factors = doc_length / X_length
        X_out = factors.reshape(1, -1) * X
        return X_out

    def _e_step(self, F, L, X):
        X_out = np.zeros((self.n_topics, F.shape[1]))
        aux3 = L.sum(axis=0) + 1 / self.gamma_scale_prior
        aux4 = (self.gamma_shape_prior - 1) / X
        cooF = sparse.coo_matrix(F)
        cooY_data = np.dot(L, X)[cooF.row, cooF.col]
        aux0 = cooF.data / cooY_data
        for i in range(self.n_topics):
            aux1 = sparse.coo_matrix(
                (aux0 * L[cooF.row, i], (cooF.row, cooF.col)),
                shape=cooF.shape)
            aux2 = aux1.sum(axis=0).A.ravel() + aux4[i, :]
            X_out[i, :] = X[i, :] * aux2 / aux3[i]
        return X_out

    def _m_step(self, F, L, X):
        L_out = np.zeros((self.n_vocab, self.n_topics))
        cooF = sparse.coo_matrix(F)
        cooY_data = np.dot(L, X)[cooF.row, cooF.col]
        aux2 = X.sum(axis=1)
        aux0 = cooF.data / cooY_data
        for j in range(self.n_topics):
            aux1 = sparse.coo_matrix(
                (aux0 * X[j, cooF.col], (cooF.row, cooF.col)), shape=cooF.shape
                                     ).sum(axis=1).A.ravel()
            L_out[:, j] = L[:, j] * aux1 / aux2[j]
        return L_out

    def fit(self, X, y=None):
        D = self.ngrams_count.fit_transform(X)
        self.vocabulary = self.ngrams_count.get_feature_names()
        self.n_samples, self.n_vocab = D.shape
        F = D.transpose()

        np.random.seed(seed=14)
        self.X_init = np.random.gamma(
            shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
            size=(self.n_topics, self.n_samples))
        self.X_init = self._rescale_X(F, self.X_init)
        np.random.seed(seed=15)
        Lambda_init = np.random.gamma(shape=1, scale=1,
                                      size=(self.n_vocab, self.n_topics))
        self.Lambda_init = self._rescale_Lambda(Lambda_init)

        X_ = self.X_init.copy()
        Lambda = self.Lambda_init.copy()

        for i in range(self.max_iters):
            Lambda_last = Lambda
            for q in range(1):
                X_ = self._e_step(F, Lambda, X_)
            Lambda = self._m_step(F, Lambda, X_)
            L_change = (
                np.linalg.norm(Lambda - Lambda_last) / np.linalg.norm(Lambda))
            rand_idx = np.random.choice(
                range(self.n_samples), size=np.minimum(1000, self.n_sample),
                replace=False)
            kl_divergence = kl_div(
                F[:, rand_idx].A, np.dot(Lambda, X_[:, rand_idx])
                ).sum() / F.shape[1]
            print('iter %d; Lambda-change: %.5f; kl_div: %.3f' %
                  (i, L_change, kl_divergence))
            if L_change < self.tol:
                break
        print('final fit iter: %d' % i)
        self.Lambda = Lambda
        self.X_ = X_
        return self

    def transform(self, X):
        D = self.ngrams_count.transform(X)
        F = D.transpose()
        X_ = np.random.gamma(
            shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
            size=(self.n_topics, D.shape[0]))
        X_ = self._rescale_X(F, X_)
        for i in range(self.max_iters):
            X_last = X_
            X_ = self._e_step(F, self.Lambda, X_)
            mean_change = self._mean_change(X_, X_last)
            if mean_change < self.tol:
                break
        # if normalize=False:
        #     X_ = X_ / X_.sum(axis=0).reshape(1, -1)
        return X_.transpose()


def batch2(iterable1, iterable2, n=1):
    len_iter = len(iterable1)
    for idx in range(0, len_iter, n):
        this_slice = slice(idx, min(idx + n, len_iter))
        yield (iterable1[this_slice],
               iterable2[this_slice])


def batch_lookup(lookup, n=1):
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield (unq_indices, indices)


def get_kmeans_prototypes(X, n_prototypes, hashing_dim=128,
                          ngram_range=(2, 4), sparse=False,
                          sample_weight=None, random_state=None):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(analyzer='char', norm=None,
                                   alternate_sign=False,
                                   ngram_range=ngram_range,
                                   n_features=hashing_dim)
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
