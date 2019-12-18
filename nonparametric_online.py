import numpy as np 
from numpy import random
import scipy.stats as stats
import os
import json
import scipy
import copy 
import scipy.sparse as sparse
import time
from scipy.special import digamma
from multiprocessing import Pool
import itertools

def simulate(U,D,K,alpha = 2, beta_shape_prior = 1, beta_rate_prior = 1, s_rate_prior = 1):
    s = np.random.gamma(shape = alpha, scale = 1/s_rate_prior, size = U)
    v = np.random.beta(a = 1, b = alpha, size = (U,K))
    theta = np.array([[s[u] * v[u,k] * np.prod(1-v[u,:k]) for k in range(K)] for u in range(U)])  
    beta = np.random.gamma(shape = beta_shape_prior, scale = 1/beta_rate_prior, size = (U,K))
    X = np.array([[np.random.poisson(theta[u,:] @ beta[d,:]) for d in range(D)] for u in range(U)])
    return X, theta, beta, s, v

class NPNMF:
    def __init__(self, X, T=512, seed=None, saved_model_file = None, **kwargs):
        self.X = X.copy()
        self.U, self.D = self.X.shape
        self.T = T
        
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
        self.alpha = float(kwargs.get('alpha', 1.1))
        self.beta_shape_prior = float(kwargs.get('beta_shape_prior', 0.3)) #a
        self.beta_rate_prior = float(kwargs.get('beta_rate_prior', 0.3))   #b          
        self.s_rate_prior = float(kwargs.get('s_rate_prior', 1.0))       #c
        
    def initialize(self, saved_model_file):
        if saved_model_file:
            self.load_model(saved_model_file)
        else:
            # variational parameters for Beta 
            self._beta_shape = np.full((self.D, self.T), 1.0)
            self._beta_rate = np.full((self.D, self.T), 1.0)
            
            # variational parameters S 
            self._s_shape = np.full(self.U, 1.0)
            self._s_rate = np.full(self.U, 1.0)
            
            # variational parameters for Z
            self._phi = np.zeros((self.U, self.D, self.T))
            #self._phi_before_normalization = np.zeros((self.U, self.D, self.T + 1)) # This is exp(E(log_theta_uk) + E(log_beta_dk))

            # variational parameters for sticks
            self._v = np.random.beta(1, self.alpha, size = (self.U, self.T))
            
        self._beta_mean = self._beta_shape /self._beta_rate #NOTE: added for caching
        self._elogbeta = digamma(self._beta_shape) - np.log(self._beta_rate) #NOTE: added for caching
        
        self._s_mean = self._s_shape/ self._s_rate #NOTE: added for caching
        self._elogs = digamma(self._s_shape) - np.log(self._s_rate) #NOTE: added for caching
            
        self._logpi = np.array([[np.log(self._v[u,k]) + np.sum(np.log(1 - self._v[u,:k])) for k in range(self.T)] for u in range(self.U)])

    def sum_logbeta_logtheta(self,u,d):
        phi = self._elogbeta[d,:] + self._elogs[u] + self._logpi[u,:] #NOTE: added for caching
        phi = np.array([*phi,self.compute_mult_normalizer_infsum(u)])
        return np.exp(phi)

    def update_phi(self):
        for u,d in self.nonzero:
            phi = self.sum_logbeta_logtheta(u,d)
            self._phi[u,d,:] = copy.deepcopy((phi/np.sum(phi))[:-1])

    def update_phi_by_u_d(self, ud):
            u,d = ud
            phi = self.sum_logbeta_logtheta(u,d)
            self._phi[u,d,:] = (phi/np.sum(phi))[:-1]
        
                       
    def elogtheta_at_truncation(self, u):
        elogsu = digamma(self._s_shape[u]) - np.log(self._s_rate[u])
        elogvt = digamma(1) - digamma(1+self.alpha)
        return elogsu + elogvt +  np.sum(np.log(1 - self._v[u,:self.T]))

    def compute_mult_normalizer_infsum(self, u):
        elogv = digamma(self.alpha) - digamma(1+self.alpha)
        return self.elogtheta_at_truncation(u) + digamma(self.beta_shape_prior) - np.log(self.beta_rate_prior) \
                       - np.log(1 - np.exp(elogv))
    
    def compute_scalar_rate_infsum(self, u):
        Y = np.exp(self._logpi[u, self.T-1] - np.log(self._v[u, self.T-1]) + np.log(1-self._v[u, self.T-1]))
        D = self.beta_shape_prior/ self.beta_rate_prior * self.D
        return Y * D

    def compute_scalar_rate_finitesum(self, u):
        return np.sum([np.exp(self._logpi[u,k]) * self.ebetasum(k) for k in range(self.T)])
        
    def update_sticks_scalars(self):
        for u in range(self.U):
            self._s_shape[u] = self.alpha + np.sum(self.X[u,:])
            self._s_rate[u] = self.s_rate_prior + self.compute_scalar_rate_infsum(u) + self.compute_scalar_rate_finitesum(u)
        self._s_mean = self._s_shape/ self._s_rate
        
    @staticmethod
    def solve_quadratic(A,B,C):
        if abs(A*C) < 1e-10 or abs(A) < 1e-10:
            if -C/B > 1e-10:
                return -C/B
            else:
                return 1e-10
        s1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2 * A)
        s2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2 * A)
        if s1 > 0.0 and s1 <= 1.0 and s2 > 0.0 and s2 <= 1.0:
            if (s1 < s2):
                return s1
            else:
                return s2
        elif (s1 > 0 and s1 <= 1):
            return s1
        elif (s2 > 0 and s2 <= 1):
            return s2
        else:
            print(f'A: {A}, B: {B}, C: {C}')
            print(f'WARNING: s1 {s1} and s2 {s2} are out of range in solve_quadratic')
            assert(0)                   
                       
    def update_sticks(self): 
        for u in range(self.U):
            #cached_prod = [np.prod(1-self._v[u,:i+1]) for i in range(self.T)]
            cached_sum = np.sum([self.X[u,d] * (1 - np.sum(self._phi[u,d,:])) for d in self.byuser[u]]) 
            for k in range(self.T):
                A = np.sum([self._v[u,l] * np.prod(1-self._v[u,:l])/(1-self._v[u,k]) * np.sum(self._beta_mean[:,l]) \
                            for l in range(k+1,self.T)]) \
                       - np.prod(1-self._v[u,:k]) * np.sum(self._beta_mean[:,k]) \
                       + self.D * self.beta_shape_prior/self.beta_rate_prior * np.prod(1-self._v[u,:]) / (1-self._v[u,k])        
                
                # A = np.sum([self._v[u,k+1:] * cached_prod[k:self.T-1]/(1-self._v[u,k]) * self._beta_mean[:,k+1:].sum(axis = 0)]) \
                #        - (1 if k == 0 else cached_prod[k-1]) * np.sum(self._beta_mean[:,k]) \
                #        + self.D * self.beta_shape_prior/self.beta_rate_prior * cached_prod[self.T-1] / (1-self._v[u,k])        
                #assert A0 == A

                A = A * self._s_mean[u]
                C = -1 * self.X[u,:] @ self._phi[u,:,k]
                C = C[0]
                B = self.alpha - 1 - C - A  + cached_sum
                try:
                    self._v[u,k] = NPNMF.solve_quadratic(A, B, C)
                except:
                    raise ValueError(f'Need to look into {u} and {k}')
        self._logpi = np.array([[np.log(self._v[u,k]) + np.sum(np.log(1 - self._v[u,:k])) for k in range(self.T)] for u in range(self.U)])
         
    def ethetasum(self,k):
        return np.sum([(self._s_mean[u] + np.exp(self._logpi[u,k])) for u in range(self.U)])
    
    def ebetasum(self,k):
        return np.sum(self._beta_mean[:,k])
                       
    def update_items(self):
        for d in range(self.D):
            for k in range(self.T):
                self._beta_shape[d,k] = self.beta_shape_prior + self.X[:,d].T @ self._phi[:,d,k]
                self._beta_rate[d,k] = self.beta_rate_prior + self.ethetasum(k)
        self._beta_mean = self._beta_shape/self._beta_rate
                
    def inference(self):
        _iter = 0
        last_ELBO = 0
        last_logjoint = 0
        while True:
            _iter += 1
            t0 = time.time()
            
            #Update phi
            self.update_phi() #q(phi_ud)
            t1 = time.time()
            #print(f'Iter {_iter}: Update phi = {t1 - t0}')
            #print(f'ElBO: {self.ELBO()}')

            #Update across user
            self.update_sticks() #q(v_uk)
            t2 = time.time()
            #print(f'Iter {_iter}: Update user stick = {t2 - t1}')
            #print(f'ElBO: {self.ELBO()}')

            self.update_sticks_scalars() #q(s_u)
            t3 = time.time()
            #print(f'Iter {_iter}: Update user stick scalar = {t3 - t2}')
            #print(f'ElBO: {self.ELBO()}')

            #Update across item
            self.update_items() #q(beta_d)
            t4 = time.time()
            #print(f'Iter {_iter}: Update items = {t4 - t3}')
            #print(f'ElBO: {self.ELBO()}')

            print(f'logjoint:{self.logjoint()}')

            # #Validate
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
            
            # #Validate
            # ELBO = self.ELBO()
            # print(f'Iter {_iter}: ELBO = {ELBO}, last_ELBO = {last_ELBO}')
            # if _iter > 1 and abs(ELBO/last_ELBO - 1) < self.threshold:
            #     print('Converged!')
            #     break
            # elif _iter >= self.max_iter:
            #     print(f'Stopped at {_iter}')
            #     break
            # else:
            #     last_ELBO = ELBO
                            
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
    
    def pair_likelihood(self, u, d, y):
        interaction = np.sum([self._s[u] * self._v[u,k] * np.prod(1-self._v[u,:k]) * self._beta_shape[d]/self._beta_rate[d] for k in range(self.T)])
        return y * np.log(interaction) - interaction - np.log(np.factorial(y))

    def logjoint(self):
        X_flatten = self.X.toarray().flatten()
        beta = self._beta_shape/self._beta_rate
        theta = np.array([[self._s_shape[u]/self._s_rate[u] * self._v[u,k] *np.prod(1-self._v[u,:k]) for k in range(self.T)] for u in range(self.U)])
        mu = (theta @ beta.T).flatten()
        mu = mu[X_flatten > 0]
        X_flatten = X_flatten[X_flatten > 0]
        assert mu.shape == X_flatten.shape, f'{mu.shape} vs {X_flatten.shape}'
        return (np.sum(X_flatten * np.log(mu)) - np.sum(mu))

    def ELBO(self):
        s = 0 
        # from x_ud.log(sum(beta_dk @ theta_uk))
        s1 = np.sum([self.X[u,d] * np.sum(self.sum_logbeta_logtheta(u,d)) for u,d in self.nonzero])

        # from sum(beta_dk * theta_uk)
        s2 = -np.sum([(self.compute_scalar_rate_infsum(u) + self.compute_scalar_rate_finitesum(u)) * self._s_shape[u]/self._s_rate[u] for u in range(self.U)])
        
        # from v
        s3 = np.sum([(self.alpha - 1) * np.log(1- self._v[u,k]) for u in range(self.U) for k in range(self.T)])
        
        # from beta
        s4 = np.sum([(self.beta_shape_prior-1) * (digamma(self._beta_shape[d,k]) - np.log(self._beta_rate[d,k])) - self.beta_rate_prior * self._beta_mean[d,k] for d in range(self.D) for k in range(self.T)])
        
        # from s
        s5 = np.sum([(self.alpha -1) * (digamma(self._s_shape[u])- np.log(self._s_rate[u])) - self.s_rate_prior * self._s_mean[u] for u in range(self.U)])
        
        # normalizer for s
        s6 = -np.sum([digamma(self._s_shape[u]) - np.log(self._s_rate[u]) for u in range(self.U)])

        # normalizer for beta
        s7 = -np.sum([digamma(self._beta_shape[d,k]) - np.log(self._beta_rate[d,k]) for d in range(self.D) for k in range(self.T)])
        
        print(f'phi: {s1 +s2} , v: {s3}, beta: {s4 + s7} ,s :{s5}')
        return s1 + s2 + s3 + s4 + s5 + s6 + s7 

     def online_inference(self, est_total=None):
        '''Fit the model to the data in X. X can be a scipy.sparse.csr_matrix
        Parameters
        ----------
        X : array-like, shape (n_items, n_users)
            Training data.
        est_total : int
            The estimated size of the entire data. Could be larger than the
            actual size.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self._scale = float(self.U) / self.batch_size
        self.bound = list()
        for count in xrange(self.n_pass):
            if self.verbose:
                print 'Iteration %d: passing through the data...' % count
            indices = np.arange(self.U)
            if self.shuffle:
                np.random.shuffle(indices)
            for (i, istart) in enumerate(xrange(0, self.U, self.batch_size)):
                print '\tMinibatch %d:' % i
                idx = indices[istart: istart + self.batch_size]
                mini_batch = self.X[idx]
                if scipy.sparse.isspmatrix(mini_batch):
                    mini_batch = mini_batch.toarray()
                self.set_learning_rate(mini_batch, iter=count * self.U / self.batch_size + i)
                self.partial_fit(mini_batch, idx=idx)
                self.bound.append(self._stoch_bound(mini_batch, idx=idx))
                
    def partial_fit(self, X, idx=None):
        '''Fit the data in X as a mini-batch and update the parameter by taking
        a natural gradient step.
        Parameters
        ----------
        X : array-like, shape (batch_size, n_feats)
            Mini-batch data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.transform(X, idx=idx)
        # take a (natural) gradient step

        ratio = X / self._xexplog(idx=idx)
        self.nu_t = (1 - self.rho) * self.nu_t + self.rho * \
            (self.a + self._scale * np.exp(self.Elogt) * np.exp(self.Elogb[idx].T).dot(ratio))
        self.rho_t = (1 - self.rho) * self.rho_t + self.rho * \
            (self.b + self._scale * self.Eb[idx].T.dot(self.Elam))
        self.Et, self.Elogt = comp_gamma_expectations(self.nu_t, self.rho_t)
        return self
 












def validate(theta,beta,rating_valid):
    size = len(list(zip(*rating_valid.nonzero())))
    rating_valid = rating_valid.toarray().flatten()
    mu = (theta @ beta.T).flatten()
    mu = mu[rating_valid > 0]
    rating_valid = rating_valid[rating_valid > 0]
    mu[mu>10] = 10
    assert mu.shape == rating_valid.shape, f'{mu.shape} vs {rating_valid.shape}'
    return (np.sum(rating_valid * np.log(mu)) - np.sum(mu))/size
    

def vi(rating_train, rating_valid, **kwargs):
    #Use sparse matrix representation
    U,D = rating_train.shape
    indices = rating_train.indices
    indptr = rating_train.indptr
    nonzero = list(zip(*rating_train.nonzero()))
    byrow = {row:[indices[i] for i in range(indptr[row], indptr[row+1])] for row in range(U)}
    
    rating_csc = rating_train.tocsc()
    indices = rating_csc.indices
    indptr = rating_csc.indptr
    bycol = {col:[indices[i] for i in range(indptr[col], indptr[col+1])] for col in range(D)}

    #Starting values    
    T = kwargs.pop('T', 50) #Truncate level
       
    kappa_rate = np.array([0.3]*U) 
    tau_rate = np.array([0.3]*D) 
    kappa_shape = a0 + K * a1
    tau_shape = m0 + K * m1
    gamma_shape = np.array([[0.3]*K]*U) 
    gamma_rate = random.gamma(shape = kappa_shape, scale = 1/0.3, size = (U,K))
    gamma_rate0 = copy.deepcopy(gamma_rate)
    lambda_shape = np.array([[0.3]*K]*D) 
    lambda_rate = random.gamma(shape = lambda_shape, scale = 1/0.3, size = (D,K)) 
    phi = np.zeros((U,D,K))
    
    #CAVI
    max_iter = kwargs.pop('max_iter', 10)
    threshold = kwargs.pop('threshold', 10e-4)
    n_iter = 0 
    
    while True:
        if n_iter == 5:
            print()
        #Update phi
        time0 = time.time()
        for u,d in nonzero:
            phi[u,d,:] = np.exp([scipy.special.digamma(gamma_shape[u,k]) - np.log(gamma_rate[u,k]) \
                    + scipy.special.digamma(lambda_shape[d,k]) - np.log(lambda_rate[d,k]) \
                        for k in range(K)])
            phi[u,d,:] = phi[u,d,:] / np.sum(phi[u,d,:])
        
        #Update gamma and kappa
        time1 = time.time()
        for u in range(U):
            gamma_shape[u,:] = a1 + rating_train[u,:] @ phi[u,:,:]
            for k in range(K):
                gamma_rate[u,k] = kappa_shape/kappa_rate[u] + np.sum([lambda_shape[d,k]/lambda_rate[d,k] for d in byrow[u]])
            kappa_rate[u] = a0/b0 + np.sum([gamma_shape[u,k]/gamma_rate[u,k] for k in range(K)])
        
        #Update lambda and tau
        time2 = time.time()
        for d in range(D):
            lambda_shape[d,:] = m1 + rating_train[:,d].T @ phi[:,d,:]
            for k in range(K):
                lambda_rate[d,k] = tau_shape/tau_rate[d] + np.sum([gamma_shape[u,k]/gamma_rate[u,k] for u in bycol[d]])
            tau_rate[d] = m0/n0 + np.sum([lambda_shape[d,k]/lambda_rate[d,k] for k in range(K)])
        
        time3 = time.time()
        print(f'Time update phi: {time1 - time0}, Time update gamma and kappa: {time2 - time1}, Time update lambda and tau: {time3 - time2}')
        
        #Validate
        theta, beta = gamma_shape/gamma_rate, lambda_shape/lambda_rate
        training_likelihood = validate(theta, beta, rating_train)
        val_likelihood = validate(theta, beta, rating_valid)
        print(f'Iter {n_iter}: training_error = {training_likelihood}, val_error = {val_likelihood}, max_pref = {np.sum(theta,0).max()}, min_pref = {np.sum(theta,0).min()},max_att = {np.sum(beta,0).max()}, min_att = {np.sum(beta,0).min()}')
        if n_iter > 0:
            if abs(val_likelihood/last_val_likelihood-1) < threshold or n_iter >= max_iter:
                print(f'Compete after {n_iter} iterations: Validation Error = {val_likelihood}')
                break
        last_val_likelihood = val_likelihood
        n_iter += 1 
    np.savez(r'C:\git\PGM\temp.npz', theta = theta, beta = beta)
    return theta, beta

def cavi():
    pass

def sgd():
    pass

def natural_gradient():
    pass

def using_edward():
    pass

def using_pystan():
    pass

def vae():
    pass

def advi():
    pass

# if __name__ == "__main__":
#     rating_train, rating_valid, rating_test, movie_map = read()
#     #rating = np.array([[random.poisson(3) for i in range(2000)] for j in range(100)])
#     gibbs(rating_valid, rating_valid,max_iter = 50 )
    
#     # model = vi(rating_valid, rating_valid, max_iter = 50)

if __name__ == "__main__":
    X, theta, beta, s, v = simulate(200, 100, 20, 1, 1, 1, 1)
    X = scipy.sparse.csr_matrix(X)
    nmf = NPNMF(X,T = 30, seed = 1, threshold = 1e-8)
    nmf.inference()
    # temp = copy.deepcopy(nmf._phi)
    # nmf = NPNMF(X,T = 30, seed = 0, threshold = 1e-8)
    #nmf.update_phi()
    # temp2 = copy.deepcopy(nmf._phi)
    print()
    