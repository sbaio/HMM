library(mvtnorm)
Test <- as.data.frame(read.table("~/Documents/Mash/ModeleGraphique/DM2/EMGaussian.test", header=FALSE))
Data <- as.data.frame(read.table("~/Documents/Mash/ModeleGraphique/DM2/EMGaussian.data", header=FALSE))

# Data

Train<-as.matrix(Data)
Test<-as.matrix(Test)


# Affect, center and normalize the data
Train<-scale(Data)
Test<-scale(Test)

# Data from the previous Homework


# Additional values needed in this homework
U <- Train

A <- matrix(rep(1/6,16), nrow=4)
diag(A) <- c(rep(1/2,4))
pi <- rep(1/4,4)
time <- 500
epsilon<-1
V <- Test

# We want to get back the mu and the matrix Sigma from our EM_algorithm

MU=Jmin(mu,Train,0.1,500)
mu0<-list(MU[[1]][1,], MU[[1]][2,], MU[[1]][3,], MU[[1]][4,])
sigma0<-list(diag(2),diag(2),diag(2),diag(2))
pi0=c(1/4,1/4,1/4,1/4)
epsilon=0.001

Train2=Train[1:time,]
# Parameters we will use
#mu_EM <- EM_GaussMixt(Train, mu0, sigma0, pi0, epsilon, Proportionnal=FALSE)$mu
mu_EM <- EM_GaussMixt(Train2, mu0, sigma0, pi0, epsilon, Proportionnal=FALSE)$mu
#mu_EM <- list(as.vector(mu_EM[[1]]),as.vector(mu_EM[[2]]),as.vector(mu_EM[[3]]),as.vector(mu_EM[[4]]))
#sigma_EM <- EM_GaussMixt(Train, mu0, sigma0, pi0, epsilon, Proportionnal=FALSE)$sigma
sigma_EM <- EM_GaussMixt(Train2, mu0, sigma0, pi0, epsilon, Proportionnal=FALSE)$sigma


## Alpha-recursion

# Name: logalpha
# Input: U the data, A the transition matrix, pi the initial probability distribution, mu & sigma the parameters of the normal
# Output: The matrix of the log alpha
logalpha <- function(U, A, pi, mu, sigma){
  
  n <- nrow(U)
  K <- length(pi)
  logalpha <- matrix(rep(0,n*K), nrow=n)
  
  # The first alpha is computed with the initial probability distribution
  for (k in 1:K){  
    logalpha[1,k] <- log(pi[k]) + log(dmvnorm(U[1,], mu[[k]], sigma[[k]]))
  }
  
  # The other are computed with the recursive formula
  # We fill our matrix logalpha row by row
  for(t in 2:n){
    
    for(k in 1:K){
      
      # We create for each k the vector a such that:
      # a[i]=log(p(qt=k|q(t-1)=i))+log(alpha(q(t-1)=i))
      # k corresponds to the current state (t)
      # i corresponds to the previous state (t-1)
      a <- rep(0,K)
      
      for(i in 1:K){
        a[i] <- logalpha[t-1,i] + log(A[k,i]) 
      }
      
      logalpha[t,k] <- log(dmvnorm(U[t,], mu[[k]], sigma[[k]]))+ max(a) + log(sum(exp(a - max(a)))) 
    }
  }
  return(logalpha)
}

## Beta-recursion

# Name: logbeta
# Input: U : the data, A the transition matrix, pi the initial probability distribution, mu & sigma the parameters of the normal
# Output: The matrix of the log beta
logbeta <- function(U, A, pi, mu, sigma){ 
  
  n <- nrow(U)
  K <- length(pi)
  logbeta <- matrix(rep(0,K*n), nrow=n)
  
  # The last beta is given by the vector (1,1,1,1)
  logbeta[n,] <- log(rep(1,K))
  
  # The other are computed with the recursive formula
  # We fill our matrix logalpha row by row
  for(t in (n-1):1){
    
    for(k in 1:K){
      
      # We create for each k the vector a such that:
      # a[i]=log(beta(q(t+1)=i))+log(p(q(t+1)=i|q(t)=k))+log(p(u(t+1)|q(t+1)=i))
      # k corresponds to the current state (t)
      # i corresponds to the following state (t+1)
      a <- rep(0,K)
      
      for(i in 1:K){
        a[i] <- logbeta[t+1,i] + log(A[i,k]) + log(dmvnorm(U[t+1,], mu[[i]], sigma[[i]]))
      }
      logbeta[t,k] <- max(a) + log(sum(exp(a - max(a))))
    }
  }
  return(logbeta)
}

##  Conditional Probability (p(qt|u1,...,uT))

# Name: prob
# Input: U the data, A the transition matrix, pi the initial probability distribution, mu & sigma the parameters of the normal
#        time is the size we want to have 
# Output: The matrix of conditional probability
prob <- function(U, A, pi, mu, sigma, time){
  
  n <- time
  K <- length(pi)
  proba <- matrix(rep(0,n*K), nrow=n)
  
  Y <- logbeta(U, A, pi, mu, sigma) + logalpha(U, A, pi, mu, sigma)
  
  for (t in 1:n){
    
    # We get rid of the row that corresponds to the max
    X <- Y[t,Y[t,]<max(Y[t,])]
    proba[t,] <- (Y[t,] - max(Y[t,]) - log(1 + sum(exp(X - max(Y[t,])))))
    # proba[t,]<- Y[t,]-max(Y[t,])-log(sum(exp(Y[t,]-max(Y[t,]))))
  }
  
  return(exp(proba))
}

# Joint Probability (p(qt,qt+1|u1,...,uT))
# Name: joint_prob
# Input: U the data, A the transition matrix, pi the initial probability distribution, mu & sigma the parameters of the normal,t the indice
joint_prob <- function(U, A, pi, mu, sigma, time){
  l <- list()
  X <- logalpha(U, A, pi, mu, sigma)
  Y <- logbeta(U, A, pi, mu, sigma)
  K <- length(pi)
  #X <- logalpha(U, A, pi, mu, sigma)[t,]
  #Y <- logbeta(U, A, pi, mu, sigma)[t+1,]
  
  for(t in 1:(time-1)){
    proba <- matrix(rep(0,K^2), nrow = K)
    Z <- X[t,] + Y[t,]
    #p<- max(Z)+log(sum(exp(Z-max(Z))))
    B <- Z[Z<max(Z)]
    p <- max(Z) + log(1 + sum(exp(B - max(Z))))
    for(i in 1:K){
      for(j in 1:K){
        proba[i,j] <- (X[t,j] + Y[t+1,i] + log(A[i,j]) + log(dmvnorm(U[t+1,], mu[[i]], sigma[[i]])) - p)
      }
    }
    l[[t]] <- exp(proba)
  }
  return(l)
}


G <- prob(Test, A, pi, mu_EM, sigma_EM, 100)
for(i in 1:4){
  plot(G[,i], type='l', col = i)
}




## EM Algorithm

#Name : log_likelihood
log_likelihood <- function(U, A, pi, mu, sigma){
  
  Y <- logbeta(U, A, pi, mu, sigma) + logalpha(U, A, pi, mu, sigma)
  
  #X <- Y[1,Y[1,]<max(Y[1,])]
  #likelihood<- max(Y[1,]) + log(1 + sum(exp(X - max(Y[1,])))) 
  likelihood3<- max(Y[1,]) + log(sum(exp(Y[1,] - max(Y[1,])))) 
  return(likelihood3)
}

#Name: pi_new
pi_new<-function(U, A, pi, mu, sigma){
  
  pinew<-as.vector(prob(U, A, pi, mu, sigma, 1))
  
  return(pinew)
}

#Name: mu_new
mu_new<-function(U, A, pi, mu, sigma,time){
  
  munew<-list()
  cond_prob<-prob(U, A, pi, mu, sigma, time)
  
  for(i in 1:length(pi)){
    div<-0
    munew_inter=rep(0,ncol(U))
    
    for(t in 1:time){
      munew_inter<-munew_inter+cond_prob[t,i]*U[t,]
      div<-cond_prob[t,i]+div
    }
    munew[[i]]<-t(as.matrix(munew_inter/div))
    #munew[[i]]<-as.vector(munew_inter/div)
  }
  
  return(munew)
}

#Name: sigma_new
sigma_new<-function(U, A, pi, mu, sigma,time){
  
  sigmanew <- list(0)
  cond_prob<-prob(U, A, pi, mu, sigma, time)
  
  for(i in 1:length(pi)){
    sig_inter<-matrix(rep(0,ncol(U)^2),ncol=ncol(U))
    div<-0
    for(t in 1:time){
      sig_inter<-sig_inter+cond_prob[t,i] * (t(U[t,]-mu[[i]])%*%(U[t,]-mu[[i]]))
      #sig_inter <- sig_inter + cond_prob[t,i] * (U[t,]-mu[[i]]) %*% t((U[t,]-mu[[i]]))
      div<-cond_prob[t,i]+div
    }
    sig_inter<-sig_inter/div
    sigmanew[[i]]<-sig_inter
  }
  
  
  return(sigmanew)
}

#Name: A_new
A_new<-function(U, A, pi, mu, sigma,time){
  
  Anew<-matrix(rep(0,length(pi)^2),ncol=4)
  cond_prob<-prob(U, A, pi, mu, sigma, time)
  join_prob<-joint_prob(U, A, pi, mu, sigma, time)
  
  for(i in 1:length(pi)){
    
    for(j in 1:length(pi)){
      A_ij<-0
      div<-0
      for(t in 1:(time-1)){
        A_ij<-A_ij+join_prob[[t]][i,j]
        div<-div+cond_prob[t,j]
      }
      Anew[i,j]<-A_ij/div
    }
  }
  return(Anew)
}

#Name : EM_HMM
#Input : the data U, the initialization mu0, sigma0, pi0, 
#the stopping criterion epsilon, 
#Output : the parameters and the latent variables after EM convergence
EM_HMM<- function(U,V,A,pi, mu, sigma,time, epsilon){      #sigma0 et mu0 de type list (liste des sigma[j])
  
  # likelihood_Train<-list(0)
  #likelihood_Test<-list(0)
  
  mu_old<-mu
  sigma_old <- sigma
  pi_old <- pi
  A_old<-A
  max_old <- log_likelihood(U, A_old, pi_old, mu_old, sigma_old)
  max_old2<- log_likelihood(V, A_old, pi_old, mu_old, sigma_old)
  
  likelihood_Train<-max_old
  likelihood_Test<-max_old2
  
  print(max_old)
  
  
  mu_news <- mu_new(U, A_old, pi_old, mu_old, sigma_old,time)
  sigma_news <-sigma_new(U, A_old, pi_old, mu_news, sigma_old,time)
  pi_news <- pi_new(U, A_old, pi_old, mu_old, sigma_old)
  A_news<-A_new(U, A_old, pi_old, mu_old, sigma_old,time)
  
  max_new <- log_likelihood(U,A_news,pi_news,mu_news,sigma_news)
  max_new2<- log_likelihood(V,A_news,pi_news,mu_news,sigma_news)
  print(max_new)
  likelihood_Train=cbind(likelihood_Train,max_new)
  likelihood_Test=cbind(likelihood_Test,max_new2)
  compteur=3
  
  while(abs(max_old-max_new)>epsilon){
    
    mu_old <- mu_news
    sigma_old <- sigma_news
    pi_old <- pi_news
    A_old <- A_news
    max_old<-max_new 
    
    mu_news <- mu_new(U, A_old, pi_old, mu_old, sigma_old,time)
    sigma_news <-sigma_new(U, A_old, pi_old, mu_news, sigma_old,time)
    pi_news <- pi_new(U, A_old, pi_old, mu_old, sigma_old)
    A_news<-A_new(U, A_old, pi_old, mu_old, sigma_old,time)
    
    max_new <- log_likelihood(U,A_news,pi_news,mu_news,sigma_news)
    max_new2<- log_likelihood(V,A_news,pi_news,mu_news,sigma_news)
    
    likelihood_Train=cbind(likelihood_Train,max_new)
    likelihood_Test=cbind(likelihood_Test,max_new2)
    compteur=compteur+1
    print(max_new)
  }
  return(list('A' = A_news, 'mu' = mu_news, 'sigma' = sigma_news, 'pi' = pi_news,like_Train=likelihood_Train,like_Test=likelihood_Test))
}


y1 <-EM_HMM(U[1:time,],V[1:time,],A,pi, mu_EM, sigma_EM,time, 0.01)

plot(as.vector(y1$like_Train),type='l',main ='train likelihood')
plot(as.vector(y1$like_Test),type='l',main='test likelihood')


# Output : the most likely sequence of labels given the observations and the parameters
viterbi <- function(U, A, pi, mu, sigma, time){
  K <- length(pi)
  V <- matrix(rep(0,K*time), ncol = K)
  for(k in 1:K){
    V[1,k] <- log(dmvnorm(U[1,], mu[[k]], sigma[[k]])) + log(pi[k])
  }
  for (t in 2:time){
    for(k in 1:K){
      i <- which.max(log(A[k,]) + V[t-1,])
      V[t,k] <- log(dmvnorm(U[t,], mu[[k]], sigma[[k]])) + log(A[k,i]) + V[t-1,i]
    }
  }
  # Traceback :
  Q <- rep(0, time)
  Q[time] <- which.max(V[time,])
  for (s in (time-1):1){
    temp <- rep(0,K)
    for(j in 1:K){
      temp[j] <- log(A[Q[s+1],j]) + V[s, j]
    }
    Q[s] <- which.max(temp)
  }
  return(Q)
}

# Output : Clusterize the points of the most likely sequence
classification <- function(U, A, pi, mu, sigma, time){
  X <- U[1:time,]
  plot(X)
  Q <- viterbi(U, A, pi, mu, sigma, time)
  for(i in 1:length(pi)){
    points(X[Q==i,], col=i+1)
  }
}

classification(Test, y1$A, y1$pi, y1$mu, y1$sigma, 100)

# Compute the marginal probability 
H <- prob(Test, y1$A, y1$pi, y1$mu, y1$sigma, 100)
# For each state k, plot the probability to be in state k in function of the time t
for(i in 1:4){
  plot(H[,i], type='l', col = i)
}

# Plot the most likely state according to the marginal probability
plot(apply(H, 1, which.max ), type='l', main='most likely state according to marginal proba')

# Plot the most likely state according to Viterbi
vit <- viterbi(Test, y1$A, y1$pi, y1$mu, y1$sigma, 100)
plot(vit, type='l', main='most likely state according to Viterbi')