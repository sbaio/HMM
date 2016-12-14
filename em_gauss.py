# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def kmeans(X, K, max_iterations=1000, tolerance=0.0000000001, nrestart=10):
    N = np.shape(X)[0]
    mus = []
    zs = []
    Js = []
    for irestart in range(nrestart):
        Jold = np.inf
        rp = np.random.permutation(N)
        mu = X[rp[0:K],:]

        for i in range(max_iterations):
            distances = np.zeros((N, K))
            for k in range(K):
                distances[:, k] = np.sqrt(np.sum((X - mu[k,:])**2,1))

            z = np.argmin(distances, 1)
            mindist = np.min(distances, 1)
            J = np.sum(mindist) / N
            if J > Jold - tolerance: break
            Jold = J
            for k in range(K):
                mu[k,:] = np.mean(X[z == k,:], 0)

        mus.append(mu)
        zs.append(z)
        Js.append(J)

    min_J = min(Js)
    id = Js.index(min_J)
    mu_kmeans = mus[id]
    z_kmeans = zs[id]
    return mu_kmeans, z_kmeans

def log_normalize(X):
    (N, d) = np.shape(X)
    a = np.max(X, 1).reshape((N,1))
    f = np.tile(a + np.log(np.sum(np.exp(X - np.tile(a, (1, d))), 1).reshape((N,1))), (1, d))
    X = X - f
    return X

def emgaussian(X, K, max_iterations=1000, tolerance=0.0000000001):
    (N, d) = np.shape(X)
    loglikold = -np.inf
    mu_kmeans, z_kmeans = kmeans(X, K)
    mus = mu_kmeans
    sigmas = []
    for k in range(K):
        sigmas.append(np.eye(d))
    pis = 1.0 / K * np.ones((K, 1))

    for i in range(max_iterations):
        print 'iteration ', i
        logtau_unnormalized = np.zeros((N, K))
        for k in range(K):
            invSigma = np.linalg.inv(sigmas[k])
            xc = (X - np.tile(mus[k,:], (N, 1)))
            eigvalues = np.linalg.eig(sigmas[k])[0]
            logtau_unnormalized[:, k] = - 0.5 * np.sum(np.dot(xc,invSigma) * xc, 1) - 0.5 * np.sum(np.log( eigvalues)) - 0.5 * d * np.log(2 * np.pi) + np.log(pis[k])
        logtau = log_normalize(logtau_unnormalized)
        tau = np.exp(logtau)

        loglik = (- np.sum(logtau.reshape(-1)*tau.reshape(-1) ) + np.sum(tau.reshape(-1)*logtau_unnormalized.reshape(-1))) / N
        if loglik < loglikold + tolerance: break
        loglikold = loglik
        print 'log-likelihood: ', loglikold

        for k in range(K):
            tau_k = np.tile(tau[:, k].reshape((N, 1)), (1, d))
            mus[k,:] = np.sum(tau_k * X, 0) / np.sum(tau[:, k])
            pis[k] = 1.0 / N * np.sum(tau[:, k])
            temp = X - np.tile(mus[k,:], (N, 1))
            sigmas[k] = 1.0 / np.sum(tau[:, k]) * np.dot(np.transpose(temp), (tau_k * temp))

    return pis, mus, sigmas, logtau, tau


if __name__ == "__main__":
    K = 4
    X = np.loadtxt('data/EMGaussian.data')

    # mu_kmeans, z_kmeans = kmeans(X, K)
    # plt.scatter(X[:,0], X[:,1], c=z_kmeans)
    # plt.show()

    pis, mus, sigmas, log_tau, tau = emgaussian(X, K)
    z_em = np.argmax(tau,1)
    plt.scatter(X[:, 0], X[:, 1], c=z_em)
    plt.show()
