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
def em_hmm(U, K, A, pis, mus, sigmas, max_iterations=1000, tolerance=0.0000000001):
    (T,d) = U.shape
    old_log_likelihood = -np.inf
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
        log_likelihood = compute_log_likelihood(logbeta, logalpha)
        print 'step ', step, ' ', log_likelihood
        if np.abs(log_likelihood - old_log_likelihood) < tolerance:
            break
        old_log_likelihood = log_likelihood

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

    return A, pis, mus, sigmas, tau

################################ MAIN ##############################################

# Plotting parameters
plt.style.use("ggplot")
all_colors = [parameter['color'] for parameter in plt.rcParams['axes.prop_cycle']]


#import the data files
train = np.loadtxt('data/EMGaussian.data')
test = np.loadtxt('data/EMGaussian.test')

#center the data
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

U = train # npoints=500 x dim=2
V = test # npoints=500 x dim=2

(_,mus,sigmas,_,_) = em_gauss.emgaussian(U,4)

A = np.ones((4,4))*(1./6) + (1./3)*np.diag(np.ones(4)) # initialize A with 1/2 on diagonal and 1/6 otherwise
num_classes = 4
pis = (1. / num_classes) * np.ones(num_classes)

T = 500 
epsilon = 1

# compute alpha and beta for all test data .. after similar preprocessing to train data
logalpha = compute_logalpha(V, A, pis, mus, sigmas)
logbeta = compute_logbeta(V, A, pis, mus, sigmas)

# smoothing proba
proba_smooting = smoothing(logbeta, logalpha)
# joint proba
joint_proba = compute_joint_prob(logbeta, logalpha, U, mus, sigmas)

# plotting
proba_smooting_sample = proba_smooting[:100,:]

# f, axarr = plt.subplots(2, 2)
#
# for i in range(2):
#     for j in range(2):
#         k = i + 2*j
#         axarr[i, j].plot(proba_smooting_sample[:, k], color=all_colors[k], linewidth=2)
#         axarr[i, j].set_title("Class "+str(k+1))
# plt.suptitle("Smoothing: p(q_t|u_1,...,u_T) for 100 first values", size=24)
# plt.show()

A, pis, mus, sigmas, tau = em_hmm(U,num_classes,A,pis,mus,sigmas)
z_em = np.argmax(tau,1)
plt.scatter(U[:, 0], U[:, 1], c=z_em)
plt.show()

#viterbi

#classification




