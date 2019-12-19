import numpy as np 
from numpy import random
import scipy.stats as stats
import os
import json
import scipy
import copy 
import scipy.sparse as sparse
import time
import seaborn as sns

class HPF:
    def __init__(self, X, T=512, K = 10, seed=None, saved_model_file = None, **kwargs):
        self.X = X.copy()
        self.U, self.D = self.X.shape
        self.T = T
        self.K = K
        
        #Working with sparse matrix
        indices = X.indices
        indptr = X.indptr
        self.nonzero = list(zip(*X.nonzero()))
        self.byuser = {row:[indices[i] for i in range(indptr[row], indptr[row+1])] for row in range(self.U)}
        
        rating_csc = X.tocsc()
        indices = rating_csc.indices
        indptr = rating_csc.indptr
        self.byitem = {col:[indices[i] for i in range(indptr[col], indptr[col+1])] for col in range(self.D)}

        self._parse_args(**kwargs)
        if seed is None:
            print('Using random seed')
            np.random.seed()
        else:
            print(f'Using fixed seed {seed}')
            np.random.seed(seed)
        self.initialize(saved_model_file)

    def _parse_args(self, **kwargs):
        '''
        Parse the hyperparameters
        '''
        self.threshold = float(kwargs.get('threshold', 1e-4))
        self.max_iter = int(kwargs.get('max_iter', 20))
        
    def initialize(self, saved_model_file):
        if saved_model_file:
            self.load_model(saved_model_file)
        else:
            self.a0,self.b0,self.a1 = 0.3,1,0.3  #global parameters for all users
            self.m0,self.n0,self.m1 = 0.3,1,0.3  #global parameters for all movies
            self.a2 = 2
            self.m2 = 2
            self._kappa_rate = np.array([0.3]*self.U) 
            self._tau_rate = np.array([0.3]*self.D) 
            self._kappa_shape = self.a0 + self.K * self.a1
            self._tau_shape = self.m0 + self.K * self.m1
            self._gamma_shape = np.array([[0.3]*self.K]*self.U) 
            self._gamma_rate = random.gamma(shape = self._kappa_shape, scale = 1/0.3, size = (self.U,self.K))
            self._lambda_shape = np.array([[0.3]*self.K]*self.D) 
            self._lambda_rate = random.gamma(shape = self._lambda_shape, scale = 1/0.3, size = (self.D,self.K)) 
            self._phi = np.zeros((self.U,self.D,self.K))
    
    def logjoint(self):
        size = len(list(zip(*self.X.nonzero())))
        X_flatten = self.X.toarray().flatten()
        theta, beta = self._gamma_shape/self._gamma_rate, self._lambda_shape/self._lambda_rate
        mu = (theta @ beta.T).flatten()
        mu = mu[X_flatten > 0]
        X_flatten = X_flatten[X_flatten > 0]
        return np.sum(X_flatten * np.log(mu)) - np.sum(mu)
    
    def inference(self, hierachial = True):
        _iter = 0
        last_logjoint = 0
        while True:
            #Update phi
            time0 = time.time()
            for u,d in self.nonzero:
                self._phi[u,d,:] = np.exp([scipy.special.digamma(self._gamma_shape[u,k]) - np.log(self._gamma_rate[u,k]) \
                        + scipy.special.digamma(self._lambda_shape[d,k]) - np.log(self._lambda_rate[d,k]) \
                            for k in range(self.K)])
                self._phi[u,d,:] = self._phi[u,d,:] / np.sum(self._phi[u,d,:])
            
            #Update gamma and kappa
            time1 = time.time()
            for u in range(self.U):
                self._gamma_shape[u,:] = self.a1 + self.X[u,:] @ self._phi[u,:,:]
                for k in range(self.K):
                    if hierachial:
                        self._gamma_rate[u,k] = self._kappa_shape/self._kappa_rate[u] + np.sum([self._lambda_shape[d,k]/self._lambda_rate[d,k] for d in self.byuser[u]])
                    else:
                        self._gamma_rate[u,k] = self.a2 + np.sum([self._lambda_shape[d,k]/self._lambda_rate[d,k] for d in self.byuser[u]])
                if hierachial:        
                    self._kappa_rate[u] = self.a0/self.b0 + np.sum([self._gamma_shape[u,k]/self._gamma_rate[u,k] for k in range(self.K)])
            
            #Update lambda and tau
            time2 = time.time()
            for d in range(self.D):
                self._lambda_shape[d,:] = self.m1 + self.X[:,d].T @ self._phi[:,d,:]
                for k in range(self.K):
                    if hierachial:
                        self._lambda_rate[d,k] = self._tau_shape/self._tau_rate[d] + np.sum([self._gamma_shape[u,k]/self._gamma_rate[u,k] for u in self.byitem[d]])
                    else:
                        self._lambda_rate[d,k] = self.m2 + np.sum([self._gamma_shape[u,k]/self._gamma_rate[u,k] for u in self.byitem[d]])
                if hierachial:
                    self._tau_rate[d] = self.m0/self.n0 + np.sum([self._lambda_shape[d,k]/self._lambda_rate[d,k] for k in range(self.K)])
            
            time3 = time.time()
            #print(f'Time update phi: {time1 - time0}, Time update gamma and kappa: {time2 - time1}, Time update lambda and tau: {time3 - time2}')
            
            #Validate
            _iter += 1
            logjoint = self.logjoint()
            print(f'Iter {_iter}: logjoint = {logjoint}, last_logjoint = {last_logjoint}')
            if _iter > 1 and abs(logjoint/last_logjoint - 1) < self.threshold:
                print('Converged!')
                break
            elif _iter >= self.max_iter:
                print(f'Stopped at {_iter}')
                break
            else:
                last_logjoint = logjoint
        self._theta, self._beta = self._gamma_shape/self._gamma_rate, self._lambda_shape/self._lambda_rate
            
            #theta, beta = self._gamma_shape/self._gamma_rate, self._lambda_shape/self._lambda_rate
            #training_likelihood = validate(theta, beta, self.X)
            #val_likelihood = validate(theta, beta, self.X)
            # print(f'Iter {n_iter}: training_error = {training_likelihood}')#', val_error = {val_likelihood}, max_pref = {np.sum(theta,0).max()}, min_pref = {np.sum(theta,0).min()},max_att = {np.sum(beta,0).max()}, min_att = {np.sum(beta,0).min()}')
            # if n_iter > 0:
            #     if abs(val_likelihood/last_val_likelihood-1) < threshold or n_iter >= max_iter:
            #         print(f'Compete after {n_iter} iterations: Validation Error = {val_likelihood}')
            #         break
            # last_val_likelihood = val_likelihood
            # n_iter += 1 
    def save_model(self, overwrite = True):
        for i in range(1,100):
            if overwrite or not os.path.exists(f'./model_{i}.npz'):
                np.savez(f'./model_{i}.npz', 
                        v = self._v, 
                        beta_shape = self._beta_shape,
                        beta_rate = self._beta_rate,
                        phi = self._phi,
                        s_shape = self._s_shape,
                        s_rate = self._s_rate)
                print(f'Save as model_{i}.npz')
                return 0
    
    def load_model(self, filename):
        loaded = np.load(filename)
        self._v = loaded['v']
        self._beta_shape = loaded['beta_shape']
        self._beta_rate = loaded['beta_rate']
        self._phi = loaded['phi']
        self._s_shape = loaded['s_shape']
        self._s_rate = loaded['s_rate'] 
    
    def posterior_check(self,user = True):
        poisson_mean = self._theta @ self._beta.T
        poisson_mean[self.X.toarray() == 0] = 0
        simulated = np.random.poisson(poisson_mean)
        #posterior check
        if user:
            ax = sns.distplot(np.sum(self.X,0),hist=True, kde=True)
            sns.distplot(np.sum(simulated,0),hist=True, kde=True)
            ax.set_title('Distribution of users by total rating')
            ax.set(xlabel='Total rating per user')
            ax.legend(['Observed Data', 'Simulated from fitted model'])
            ax.plot()
        else:
            ax = sns.distplot(np.sum(self.X,1),hist=True, kde=True)
            sns.distplot(np.sum(simulated,1),hist=True, kde=True)
            ax.set_title('Distribution of items by total rating')
            ax.set(xlabel='Total rating per item')
            ax.legend(['Observed Data', 'Simulated from fitted model'])
            ax.plot()
       