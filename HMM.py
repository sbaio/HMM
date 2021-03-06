# encoding: utf-8

import numpy as np
from sklearn import preprocessing
from scipy.stats import multivariate_normal
import em_gauss

import matplotlib.pyplot as plt

def logsumexp(x):
    return np.log(np.sum(np.exp(x)))

# sum calculated with log sum exponential - not yet used
def sum_lse(x):
    a = np.log(x)
    return np.exp(np.max(a) + np.log(np.sum(np.exp(a-np.max(a)))))

## alpha recursion
# A is the transition matrix
# pi the initial probability distribution
# mu & sigma the parameters of the normal
# mu is a matrix of K rows and 2 columns
# sigma is a list of K covariance matrices
# compute_logalpha returns a (T,K) matrix
def compute_logalpha(U, A, pi, mu, sigma):
    (T,d) = U.shape
    K = len(pi) # number of classes
    logalpha = np.zeros((T,K))

    # first alpha is computed with initial proba distribution
    for k in range(0,K):
        mvnU = multivariate_normal.pdf(U[0,:], mean=mu[k,:], cov=sigma[k])
        logalpha[0,k] = np.log(pi[k]) + np.log(mvnU)

    for t in range(1,T):
        for k in range(0,K):
            # We create for each k the vector a such that:
            # a[i]=log(p(qt=k|q(t-1)=i))+log(alpha(q(t-1)=i))
            # k corresponds to the current state (t)
            # i corresponds to the previous state (t-1)
            a = logalpha[t-1,:] + np.log(A[k,:])
            mvnU = multivariate_normal.pdf(U[t,:], mean=mu[k,:], cov=sigma[k])
            logalpha[t,k] = np.log(mvnU) + np.max(a) + logsumexp(a-np.max(a))

    return logalpha

## beta recursion
# compute_logbeta returns a (T,K) matrix
def compute_logbeta(U, A, pi, mu, sigma):
    (T,d) = U.shape
    K = len(pi) # number of classes
    logbeta = np.zeros((T,K))

    # The last beta is given by the vector (1,1,1,1)
    logbeta[T-1,:] = np.log(np.ones((1,K)))

    # The other are computed with the recursive formula
    # We fill our matrix logbeta row by row
    for t in range(T-2,-1,-1):
        for k in range(0,K):
            # We create for each k the vector a such that:
            # a[i]=log(beta(q(t+1)=i))+log(p(q(t+1)=i|q(t)=k))+log(p(u(t+1)|q(t+1)=i))
            # k corresponds to the current state q(t)
            # i corresponds to the following state q(t+1)
            a = np.zeros(K)
            for i in range(0,K):
                mvnU = multivariate_normal.pdf(U[t+1,:], mean=mu[i,:], cov=sigma[i])
                a[i] = logbeta[t+1,i] + np.log(A[i,k]) + np.log(mvnU)
            logbeta[t,k] = np.max(a) + logsumexp(a-np.max(a))

    return logbeta

## computes log p(u_1, ..., u_T)
def compute_log_likelihood(logbeta, logalpha):
    # We take for example q_0 to compute the probability by marginalization
    a = (logalpha + logbeta)[0, :]
    log_likelihood = np.max(a) + logsumexp(a - np.max(a))
    return log_likelihood

## Smoothing p(q_t|u_1,...,u_T) returns matrix of size T x K
## proba that u_t belongs to class k, given all observations
def smoothing(logbeta, logalpha):
    log_nom = logbeta + logalpha
    log_denom = compute_log_likelihood(logbeta, logalpha)
    return np.exp(log_nom-log_denom)

# Joint Probability (p(qt,qt+1|u1,...,uT))
def compute_joint_prob(logbeta, logalpha, U, mu, sigma):
    (n,K) = logalpha.shape
    proba = np.zeros((n-1,K,K))
    log_denom = compute_log_likelihood(logbeta, logalpha)

    for t in range(0,n-1):
        # calculate the log_denominator term with log-sum-exp trick
        log_prob_t = np.zeros((K,K))
        # i loops over values taken by q_t+1
        for i in range(K):
            mvnU = multivariate_normal.pdf(U[t+1,:], mean=mu[i,:], cov=sigma[i])
            # j loops over values taken by q_t
            for j in range(K):
                log_prob_t[i,j] = logalpha[t, j] + logbeta[t + 1, i] + np.log(A[i, j]) + np.log(mvnU)
        proba[t] = np.exp(log_prob_t - log_denom)

    return proba

# We implement EM for HMM
# The initial parameters are given using the EM-gaussian algorithm
# U are the train data
# V are the test data
# We learn from U and we test on V
def em_hmm(U, V, K, A, pis, mus, sigmas, max_iterations=1000, tolerance=10e-6):
    (T,d) = U.shape
    old_log_likelihood = -np.inf
    log_likelihood = []
    log_likelihood_test = []
    for step in range(max_iterations):
        # E - step
        logalpha = compute_logalpha(U, A, pis, mus, sigmas)
        logbeta = compute_logbeta(U, A, pis, mus, sigmas)

        tau = smoothing(logbeta, logalpha)
        tau_transition = compute_joint_prob(logbeta, logalpha, U, mus, sigmas)
        # At each t
        # the rows of tau_transition[t,:,:] are for q_t+1
        # the columns are for q_t

        # log-likelihood
        log_likelihood.append(compute_log_likelihood(logbeta, logalpha))
        # print 'step ', step, ' ', log_likelihood[-1]
        # We compute the log_likelihood for the test data
        logalpha_test = compute_logalpha(V, A, pis, mus, sigmas)
        logbeta_test = compute_logbeta(V, A, pis, mus, sigmas)
        log_likelihood_test.append(compute_log_likelihood(logbeta_test, logalpha_test))

        if np.abs(log_likelihood[-1] - old_log_likelihood) < tolerance:
            break
        old_log_likelihood = log_likelihood[-1]

        # M - step
        # We update pi
        pis = tau[0,:]
        sum_tau = np.sum(tau, axis=0).reshape((K,1))
        sum_tau_transition = np.sum(tau_transition, axis=0)
        # We update A
        for i in range(K): #i loops for q_t+1
            for j in range(K): #j loops for q_t
                A[i,j] = sum_tau_transition[i,j] / sum_tau[j]
            # We update mus
            tau_u = np.dot(tau[:,i], U)
            mus[i,:] = tau_u / sum_tau[i]
            # We update sigmas
            U_centered = U - mus[i,:]
            sigmas[i] = np.zeros((d,d))
            for t in range(T):
                Uc_t = U_centered[t,:].reshape((d,1))
                covariance = np.dot(Uc_t, Uc_t.T)
                sigmas[i] += tau[t,i]*covariance
            sigmas[i] = sigmas[i] / sum_tau[i]

    return A, pis, mus, sigmas, tau, log_likelihood, log_likelihood_test

def viterbi(U,A,pi,mu,sigma):
    (T,d) = U.shape
    K = len(pi)
    V = np.zeros((T,K))

    # Initialise
    for k in range(0,K):
        mvnU = multivariate_normal.pdf(U[0,:], mean=mu[k,:], cov=sigma[k])
        V[0,k] = np.log(pi[k]) + np.log(mvnU)

    # Iterate
    for t in range(1,T):
        for k in range(0,K):
            i = np.argmax(np.log(A[k,:]) + V[t-1,:])
            mvnU = multivariate_normal.pdf(U[t,:], mean=mu[k,:], cov=sigma[k])
            V[t,k] = np.log(mvnU) + np.log(A[k,i]) + V[t-1,i]

    # Traceback
    q = np.zeros(T,dtype=int)
    q[T-1] = np.argmax(V[T-1,:])

    for s in range(T-2,-1,-1):
        temp = np.zeros(K)
        for j in range(0,K):
            temp[j] = np.log(A[q[s+1],j]) + V[s,j]

        q[s] = np.argmax(temp)

    return q

################################ MAIN ####################################################################

if __name__ == "__main__":

    ## Plotting parameters
    plt.style.use("ggplot")
    all_colors = [parameter['color'] for parameter in plt.rcParams['axes.prop_cycle']]


    ## import the data files
    train = np.loadtxt('data/EMGaussian.data')
    test = np.loadtxt('data/EMGaussian.test')

    ## center the data
    # scaler = preprocessing.StandardScaler().fit(train)
    # train = scaler.transform(train)
    # test = scaler.transform(test)

    U = train # npoints=500 x dim=2
    V = test # npoints=500 x dim=2
    K = 4

    ## initialization of the parameters
    (_,mus,sigmas,log_likelihood_gauss,log_likelihood_gauss_test,_) = em_gauss.em_gaussian(U, V, K)

    A = np.ones((K,K))*(1./6) + (1./3)*np.diag(np.ones(4)) # initialize A with 1/2 on diagonal and 1/6 otherwise
    pis = (1. / K) * np.ones(K)

    T = 500
    epsilon = 1

    ## compute alpha and beta for all test data .. after similar preprocessing to train data
    logalpha = compute_logalpha(V, A, pis, mus, sigmas)
    logbeta = compute_logbeta(V, A, pis, mus, sigmas)

    ## smoothing proba
    proba_smoothing = smoothing(logbeta, logalpha)

    ## joint proba
    joint_proba = compute_joint_prob(logbeta, logalpha, V, mus, sigmas)

    ## plotting
    proba_smoothing_sample = proba_smoothing[:100, :]

    f, axarr = plt.subplots(2, 2)

    for i in range(2):
        for j in range(2):
            k = i + 2*j
            axarr[i, j].plot(proba_smoothing_sample[:, k], color=all_colors[k], linewidth=2)
            axarr[i, j].set_title("Class "+str(k+1))
            axarr[i, j].set_ylim([0,1.0])
    plt.suptitle("Smoothing: p(q_t|u_1,...,u_T) for 100 first values", size=24)
    plt.show()

    ## EM algorithm
    A, pis, mus, sigmas, tau, log_likelihood, log_likelihood_test = em_hmm(U, V, K, A, pis, mus, sigmas)
    print "The log_likelihoods for both GMM and HMM are:"
    print "GMM train: ", log_likelihood_gauss, " GMM test: ", log_likelihood_gauss_test
    print "HMM train: ", log_likelihood[-1], " HMM test: ", log_likelihood_test[-1]
    plt.figure()
    plt.plot(log_likelihood, label="Train data")
    plt.plot(log_likelihood_test, label="Test data")
    plt.title("Evolution of log_likelihood vs. iterations", size=18)
    plt.xlabel("Iterations", size=18)
    plt.ylabel("Log-likelihood", size=18)
    plt.legend()
    plt.show()

    ## viterbi - most likely sequence
    q_train = viterbi(U, A, pis, mus, sigmas)

    fig, ax = plt.subplots()
    markers = ['o', 's', 'D', '^']
    for k in range(K):
        # Train data
        ax.scatter(U[q_train == k, 0], U[q_train == k, 1], marker=markers[k], color=all_colors[k], s=50, alpha=0.8)
        ax.scatter(mus[k, 0], mus[k, 1], marker=markers[k], color='k', s=100)
        ax.set_title("Clustering of the train data", size=24)
    plt.show()

    ## plot p(q_t | u_1,..., u_T)
    logalpha = compute_logalpha(V, A, pis, mus, sigmas)
    logbeta = compute_logbeta(V, A, pis, mus, sigmas)
    proba_smoothing = smoothing(logbeta, logalpha)
    proba_smoothing_sample = proba_smoothing[:100, :]

    f, axarr = plt.subplots(2, 2)

    for i in range(2):
        for j in range(2):
            k = i + 2*j
            axarr[i, j].plot(proba_smoothing_sample[:, k], color=all_colors[k], linewidth=2)
            axarr[i, j].set_title("Class "+str(k+1))
            axarr[i, j].set_ylim([0,1.0])
    plt.suptitle("Smoothing: p(q_t|u_1,...,u_T) for 100 first values", size=24)
    plt.show()

    ## Most likely state
    states = np.argmax(proba_smoothing_sample, axis=1)
    plt.plot(range(100), states + 1)
    plt.ylim((0.9, 4.1))
    plt.title("Most likely state for the first 100 samples", size=24)

    ## viterbi for the test data
    q_test = viterbi(V, A, pis, mus, sigmas)
    plt.plot(range(100), q_test[:100] + 1)
    plt.ylim((0.9, 4.1))
    plt.title("Most likely state for the first 100 samples with Viterbi algorithm", size=24)

    print "% of difference in states: ", np.sum(q_test[:100] != states)/100, '%'
