# encoding: utf-8

import numpy as np
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import em_gauss

#import the data files
data = np.loadtxt('data/EMGaussian.data')
test = np.loadtxt('data/EMGaussian.test')

#center the data
train = preprocessing.scale(data)
test = preprocessing.scale(test)

U = train # npoints=500 x dim=2
V = test # npoints=500 x dim=2

(a,b,c,d,e) = em_gauss.emgaussian(U,4)

A = np.ones((4,4))*(1./6) + (1./3)*np.diag(np.ones(4)) # initialize A with 1/2 on diagonal and 1/6 otherwise
num_classes = 4
pi = (1./num_classes)*np.ones(num_classes)

T = 500 
epsilon = 1

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

		# i loops over values taken by z(t+1)
		for i in range(1,K):
			mvnU = multivariate_normal.pdf(U[t+1,:], mean=mu[i,:], cov=sigma[i])
			# j loops over values taken by z(t)
			for j in range(1,K):
				log_prob_t[i,j] = X[t,j] + Y[t+1,i] + np.log(A[i,j]) + np.log(mvnU) - log_denom_t

		proba[t] = np.exp(log_prob_t)

	return proba
				




#log_likelihood

#pi_new

#mu_new

#sigma_new

#A_new

#EM_HMM

#viterbi

#classification




