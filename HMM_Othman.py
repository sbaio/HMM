# encoding: utf-8

import numpy as np
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import em_gauss

import matplotlib.pyplot as plt

def logsumexp(x):
    return np.log(np.sum(np.exp(x)))

# sum calculated with log sum exponential - not yet used
def sum_lse(x):
    a = np.log(x)
    return np.exp(np.max(a) + np.log(np.sum(np.exp(a-np.max(a)))))

## alpha recursion
# Name : log alpha
# Input : U the data, A the transition matrix, pi the initial probability distribution, mu & sigma the parameters of the normal
# Output: The matrix of the log alpha

# mu is a matrix of K rows and 2 columns
# sigma is a list of K covariance matrices

def logalpha(U, A, pi, mu, sigma):
    n = len(U) # number of rows , data points
    K = len(pi) # number of classes
    logalpha = np.zeros((n,K))

    # first alpha is computed with initial proba distribution
    for k in range(0,K):
        mvnU = multivariate_normal.pdf(U[0,:], mean=mu[k,:], cov=sigma[k])
        logalpha[0,k] = np.log(pi[k]) + np.log(mvnU)

    for t in range(1,n):
        for k in range(0,K):
            # We create for each k the vector a such that:
            # a[i]=log(p(qt=k|q(t-1)=i))+log(alpha(q(t-1)=i))
            # k corresponds to the current state (t)
            # i corresponds to the previous state (t-1)

            a = logalpha[t-1,:] + np.log(A[k,:])

            mvnU = multivariate_normal.pdf(U[t,:], mean=mu[k,:], cov=sigma[k])
            logalpha[t,k] = np.log(mvnU) + np.max(a) + logsumexp(a-np.max(a))
    return logalpha


def logbeta(U,A,pi,mu,sigma):
    n = len(U) # number of rows , data points
    K = len(pi) # number of classes
    logbeta = np.zeros((n,K))

    # The last beta is given by the vector (1,1,1,1)
    logbeta[n-1,:] = np.log(np.ones((1,K)))

    # The other are computed with the recursive formula
    # We fill our matrix logbeta row by row
    for t in range(n-2,-1,-1):
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


##  Smoothing p(qt|u1,...,uT) returns matrix of size T x K .. proba that ut belongs to class k, given all observations
def smoothing(logbeta, logalpha):

    log_nom = logbeta + logalpha
    (n,K) = log_nom.shape
    a = log_nom-np.max(log_nom,axis=1).reshape((n,1))
    log_denom = np.max(log_nom,axis=1) + np.log(np.sum(np.exp(a),axis=1))

    return np.exp(log_nom-log_denom.reshape((n,1)))

# Joint Probability (p(qt,qt+1|u1,...,uT))
def joint_prob(logbeta, logalpha, U, mu, sigma):

    X = logalpha
    Y = logbeta
    (n,K) = X.shape

    proba = np.zeros((n-1,K,K))

    for t in range(0,n-1):
        # calculate the log_denominator term with log-sum-exp trick
        Z = X[t,:] + Y[t,:]
        log_denom_t = np.max(Z) + logsumexp(Z-np.max(Z))

        log_prob_t = np.zeros((K,K))

        # i loops over values taken by q(t+1)
        for i in range(1,K):
            mvnU = multivariate_normal.pdf(U[t+1,:], mean=mu[i,:], cov=sigma[i])
            # j loops over values taken by q(t)
            for j in range(1,K):
                log_prob_t[i,j] = X[t,j] + Y[t+1,i] + np.log(A[i,j]) + np.log(mvnU) - log_denom_t

        proba[t] = np.exp(log_prob_t)

    return proba

# We implement EM for HMM
# The initial parameters are given using the EM-gaussian algorithm
def em_hmm(U, K, A, pis, mus, sigmas, max_iterations=1000, tolerance=0.000000000):
    (n,d) = U.shape
    old_expected_lklhood = -np.inf
    for step in range(max_iterations):
        # E - step
        alpha = logalpha(U,A,pis,mus,sigmas)
        beta = logbeta(U, A, pis, mus, sigmas)

        tau = smoothing(beta, alpha)
        tau_transition = joint_prob(beta, logalpha, U, mus, sigmas)
        # At each t
        # the rows of tau_transition[t,:,:] are for q_t+1
        # the columns are for q_t

        # Expected complete log-likelihood
        expected_lklhood = np.sum(tau[0,:]*np.log(pis))
        for t in range(n):
            expected_lklhood += np.sum(tau_transition[t,:,:]*A)
            for i in range(K):
                mvnU = multivariate_normal.pdf(U[t, :], mean=mus[i, :], cov=sigmas[i])
                expected_lklhood += tau[t,i]*np.log(mvnU)
        print 'step ', step, ' ', expected_lklhood
        if np.abs(expected_lklhood - old_expected_lklhood) < tolerance:
            break
        old_expected_lklhood = expected_lklhood

        # M - step
        # We update pi
        pis = tau[0,:]
        sum_tau = np.sum(tau, axis=0)
        sum_tau_transition = np.sum(tau_transition, axis=0)
        # We update A
        for i in range(K): #i loops for q_t+1
            for j in range(K): #j loops for q_t
                A[i,j] = sum_tau_transition[i,j] / sum_tau[j]
            # We update mus
            tau_u = np.dot(tau[i].T,U)
            mus[i,:] = tau_u / sum_tau[i]
            # We update sigmas
            U_centered = U - mus[i,:]
            sigmas[i] = np.zeros((d,d))
            for t in range(T):
                covariance = np.dot(U_centered.T, U_centered)
                sigmas[i] += tau[t,i]*covariance
            sigmas[i] = sigmas[i] / sum_tau[i]

    return A, pis, mus, sigmas



# Plotting parameters
plt.style.use("ggplot")
all_colors = [parameter['color'] for parameter in plt.rcParams['axes.prop_cycle']]


#import the data files
data = np.loadtxt('data/EMGaussian.data')
test = np.loadtxt('data/EMGaussian.test')

#center the data
scaler = preprocessing.StandardScaler().fit(data)
train = scaler.transform(data)
test = scaler.transform(test)

U = train # npoints=500 x dim=2
V = test # npoints=500 x dim=2

(_,mus,sigmas,_,_) = em_gauss.emgaussian(U,4)

A = np.ones((4,4))*(1./6) + (1./3)*np.diag(np.ones(4)) # initialize A with 1/2 on diagonal and 1/6 otherwise
num_classes = 4
pi = (1./num_classes)*np.ones(num_classes)

T = 500 
epsilon = 1

# compute alpha and beta for all test data .. after similar preprocessing to train data
logalpha = logalpha(V,A,pi,mus,sigmas)
logbeta = logbeta(V,A,pi,mus,sigmas)

# smoothing proba
proba_smooting = smoothing(logbeta, logalpha)

# joint proba
joint_proba = joint_prob(logbeta, logalpha, U, mus, sigmas)

# plotting
proba_smooting_sample = proba_smooting[:100,:]

f, axarr = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        k = i + 2*j
        axarr[i, j].plot(proba_smooting_sample[:, k], color=all_colors[k], linewidth=2)
        axarr[i, j].set_title("Class "+str(k+1))
plt.suptitle("Smoothing: p(q_t|u_1,...,u_T) for 100 first values", size=24)
plt.show()

#log_likelihood

#pi_new

#mu_new

#sigma_new

#A_new

#EM_HMM

#viterbi

#classification




