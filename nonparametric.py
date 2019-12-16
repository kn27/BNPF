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
    def __init__(self, X, T=512, seed=None, **kwargs):
        '''
        BN = LVI_BP_NMF(X, K=512, smoothness=100, seed=None, alpha=2.,
                        a0=1., b0=1., c0=1e-6, d0=1e-6)
        Required arguments:
            X:              U-by-D nonnegative matrix (numpy.ndarray)
                            the data to be factorized
                            Assume scipy sparse matrix format
        Optional arguments:
            K:              the size of the initial dictionary
                            will be truncated to a proper size
            seed:           the random seed to control the random
                            initialization
                            **variational inference can only converge to local
                            optimum, thus try different seeds**
            alpha:          hyperparameter for activation.
            a0, b0:         both must be specified
                            hyperparameters for sparsity
            c0, d0:         both must be specified
                            hyperparameters for Gaussian noise
        '''
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
        self.initialize()

    def _parse_args(self, **kwargs):
        '''
        Parse the hyperparameters
        '''
        self.threshold = float(kwargs.get('threshold', 1e-4))
        self.max_iter = int(kwargs.get('max_iter', 20))
        self.alpha = float(kwargs.get('alpha', 2.))
        self.beta_shape_prior = float(kwargs.get('beta_shape_prior', 1.)) #a
        self.beta_rate_prior = float(kwargs.get('beta_rate_prior', 1.))   #b          
        self.s_rate_prior = float(kwargs.get('s_rate_prior', 0.1))       #c
        
    def initialize(self):
        # variational parameters for Beta 
        self._beta_shape = np.full((self.D, self.T), 0.3)
        self._beta_rate = np.full((self.D, self.T), 0.3)
        self._beta_mean = self._beta_shape /self._beta_rate #NOTE: added for caching
        self._elogbeta = digamma(self._beta_shape) - np.log(self._beta_rate) #NOTE: added for caching
        
        # variational parameters S 
        self._s_shape = np.full(self.U, 0.3)
        self._s_rate = np.full(self.U, 0.3)
        self._s_mean = self._s_shape/ self._s_rate #NOTE: added for caching
        self._elogs = digamma(self._s_shape) - np.log(self._s_rate) #NOTE: added for caching
        
        # variational parameters for Z
        self._phi = np.zeros((self.U, self.D, self.T))
        #self._phi_before_normalization = np.zeros((self.U, self.D, self.T + 1)) # This is exp(E(log_theta_uk) + E(log_beta_dk))

        # variational parameters for sticks
        self._v = np.random.beta(1, self.alpha, size = (self.U, self.T))

    def sum_logbeta_logtheta(self,u,d):
        #elogbeta = [digamma(self._beta_shape[d][k]) - np.log(self._beta_rate[d][k]) for k in range(self.T)]
        #elogs = digamma(self._s_shape[u]) - np.log(self._s_rate[u])
        #phi =  [elogbeta[k] + elogs + self.logpi(u,k) for k in range(self.T)]
        phi = self._elogbeta[d,:] + self._elogs[u] + np.array([self.logpi(u,k) for k in range(self.T)]) #NOTE: added for caching
        phi = np.array([*phi,self.compute_mult_normalizer_infsum(u)])
        return np.exp(phi)

    def update_phi(self):
        for u,d in self.nonzero:
            phi = self.sum_logbeta_logtheta(u,d)
            self._phi[u,d,:] = (phi/np.sum(phi))[:-1]

    def update_phi_by_u_d(self, ud):
            u,d = ud
            phi = self.sum_logbeta_logtheta(u,d)
            self._phi[u,d,:] = (phi/np.sum(phi))[:-1]
        
    def update_phi_threaded(self):
        pool = Pool(processes = 4)        
        pool.map(self.update_phi_by_u_d, self.nonzero)
        
    def logpi(self,u,k):
        return np.log(self._v[u,k]) + np.sum(np.log(1 - self._v[u,:k]))
                       
    def elogtheta_at_truncation(self, u):
        elogsu = digamma(self._s_shape[u]) - np.log(self._s_rate[u])
        elogvt = digamma(1) - digamma(1+self.alpha)
        return elogsu + elogvt +  np.sum(np.log(1 - self._v[u,:self.T]))

    def compute_mult_normalizer_infsum(self, u):
        elogv = digamma(self.alpha) - digamma(1+self.alpha)
        return self.elogtheta_at_truncation(u) + digamma(self.beta_shape_prior) - np.log(self.beta_rate_prior) \
                       - np.log(1 - np.exp(elogv))
    
    def compute_scalar_rate_infsum(self, u):
        Y = np.exp(self.logpi(u, self.T-1)) / self._v[u, self.T-1] * (1-self._v[u, self.T-1])
        D = self.beta_shape_prior/ self.beta_rate_prior * self.D
        return Y * D

    def compute_scalar_rate_infsum_check(self, u):
        Y = np.exp(self.logpi(u, self.T-1)) / self._v[u, self.T-1] * (1-self._v[u, self.T-1])
        assert np.isclose(Y,np.prod(1-self._v[u,:]))
                    
    def compute_scalar_rate_finitesum(self, u):
        return np.sum([np.exp(self.logpi(u,k)) * self.ebetasum(k) for k in range(self.T)])
        
    def update_sticks_scalars(self):
        for u in range(self.U):
            self._s_shape[u] = self.alpha + np.sum(self.X[u,:])
            self._s_rate[u] = self.s_rate_prior + self.compute_scalar_rate_infsum(u) + self.compute_scalar_rate_finitesum(u)
        self._s_mean = self._s_shape/ self._s_rate
            #self.ELBO()
        
    @staticmethod
    def solve_quadratic(A,B,C):
        if A*(-C) < 1e-10:
            return -C/B if -C/B > 1e-10 else 1e-10
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
            for k in range(self.T):
                A = np.sum([self._v[u,l] * np.prod(1-self._v[u,:l])/(1-self._v[u,k]) * np.sum(self._beta_mean[:,l]) for l in range(k+1,self.T)]) \
                       - np.prod(1-self._v[u,:k]) * np.sum(self._beta_mean[:,k]) \
                       + self.D * self.beta_shape_prior/self.beta_rate_prior * np.prod(1-self._v[u,:]) / self._v[u,k]        
                A = A * self._s_mean[u]
                C = -1 * self.X[u,:] @ self._phi[u,:,k]
                C = C[0]
                B = self.alpha - 1 - C - A  + np.sum([self.X[u,d] * (1 - np.sum(self._phi[u,d,:])) for d in range(self.D)]) 
                try:
                    self._v[u,k] = NPNMF.solve_quadratic(A, B, C)
                except:
                    raise ValueError(f'Need to look into {u} and {k}')
                    
    
    def update_sticks_threaded(self): 
        pool = Pool(processes = 12)
        def solve_quadratic_by_u_k(u,k):
            A = np.sum([self._v[u,l] * np.prod(1-self._v[u,:l])/(1-self._v[u,k]) * np.sum(self._beta_shape[:,l]/ self._beta_rate[:,l]) for l in range(k,self.T)]) \
                    - np.prod(1-self._v[u,:k]) * np.sum(self._beta_shape[:,k]/self._beta_rate[:,k]) \
                    + self.D * self.beta_shape_prior/self.beta_rate_prior * np.prod(1-self._v[u,:]) / self._v[u,k]        
            A = A * self._s_shape[u]/self._s_rate[u]
            C = -1 * self.X[u,:] @ self._phi[u,:,k]
            C = C[0]
            B = self.alpha - 1 - C - A  + np.sum([self.X[u,d] * (1 - np.sum(self._phi[u,d,:])) for d in range(self.byuser[u]])]) 
            self._v[u,k] = NPNMF.solve_quadratic(A, B, C)
        pool.map_async(solve_quadratic_by_u_k, itertools.product(range(self.U), range(self.T)) )

    def get_ABC(self):
        A = np.zeros((self.U,self.T))
        B = np.zeros((self.U,self.T))
        C = np.zeros((self.U,self.T))
        for u in range(self.U):
            for k in range(self.T):
                A[u,k] = np.sum([self._v[u,l] * np.prod(1-self._v[u,:l])/(1-self._v[u,k]) * np.sum(self._beta_mean[:,l]) for l in range(k,self.T)]) \
                       - np.prod(1-self._v[u,:k]) * np.sum(self._beta_mean[:,k]) \
                       + self.D * self.beta_shape_prior/self.beta_rate_prior * np.prod(1-self._v[u,:]) / self._v[u,k]        
                A[u,k] = A[u,k] * self._s_mean[u]
                C[u,k] = (-1 * self.X[u,:] @ self._phi[u,:,k])[0]
                B[u,k] = self.alpha - 1 - C[u,k] - A[u,k]  + np.sum([self.X[u,d] * (1 - np.sum(self._phi[u,d,:])) for d in range(self.D)]) 
        return A,B,C
         
    def ethetasum(self,k):
        return np.sum([(self._s_mean[u] + np.exp(self.logpi(u,k))) for u in range(self.U)])
    
    def ebetasum(self,k):
        return np.sum(self._beta_mean[:,k])
                       
    def update_items(self):
        for d in range(self.D):
            for k in range(self.T):
                self._beta_shape[d,k] = self.beta_shape_prior + self.X[:,d].T @ self._phi[:,d,k]
                self._beta_rate[d,k] = self.beta_rate_prior + self.ethetasum(k)
        self._beta_mean = self._beta_shape/self._beta_rate
                
    def inference(self):
        self.initialize()
        _iter = 0
        last_ELBO = 0
        while True:
            _iter += 1
            t0 = time.time()
            
            #Update phi
            self.update_phi() #q(phi_ud)
            t1 = time.time()
            print(f'Iter {_iter}: Update phi = {t1 - t0}')

            #Update across user
            self.update_sticks() #q(v_uk)
            t2 = time.time()
            print(f'Iter {_iter}: Update user stick = {t2 - t1}')

            self.update_sticks_scalars() #q(s_u)
            t3 = time.time()
            print(f'Iter {_iter}: Update user stick scalar = {t3 - t2}')

            #Update across item
            self.update_items() #q(beta_d)
            t4 = time.time()
            print(f'Iter {_iter}: Update items = {t4 - t3}')
            
            #Validate
            ELBO = self.ELBO()
            print(f'Iter {_iter}: ELBO = {ELBO}, last_ELBO = {last_ELBO}')
            if _iter > 1 and abs(ELBO/last_ELBO - 1) < self.threshold:
                print('Converged!')
                break
            elif _iter > self.max_iter:
                print(f'Stopped at {_iter}')
                break
            else:
                last_ELBO = ELBO
                            
    def save_model(self):
        for i in range(1,100):
            if not os.path.exist(f'./model_{i}.npz'):
                np.savez(f'./model_{i}.npz', 
                        v = self._v, 
                        beta_shape = self._beta_shape,
                        beta_rate = self._beta.rate,
                        phi = self._phi,
                        s_shape = self._s_shape,
                        s_rate = self._s_rate)
                return 0
    
    def load_model(self, filename):
        loaded = np.load(filename)
        self._v = loaded['v']
        self._beta_shape = loaded['beta_shape']
        self._beta.rate = loaded['beta_rate']
        self._phi = loaded['phi']
        self._s_shape = loaded['s_shape']
        self._s_rate = loaded['s_rate'] 
    
    def pair_likelihood(self, u, d, y):
        interaction = np.sum([self._s[u] * self._v[u,k] * np.prod(1-self._v[u,:k]) * self._beta_shape[d]/self._beta_rate[d] for k in range(self.T)])
        return y * np.log(interaction) - interaction - np.log(np.factorial(y))

    def ELBO(self):
        s = 0 
        # from x_ud.log(sum(beta_dk @ theta_uk))
        s1 = np.sum([self.X[u,d] * np.sum(self.sum_logbeta_logtheta(u,d)) for u,d in self.nonzero])

        # from v
        s2 = np.sum([(self.alpha - 1) * np.log(1- self._v[u,k]) for u in range(self.U) for k in range(self.T)])

        # from sum(beta_dk * theta_uk)
        s3 = -np.sum([(self.compute_scalar_rate_infsum(u) + self.compute_scalar_rate_finitesum(u)) * self._s_shape[u]/self._s_rate[u] for u in range(self.U)])

        # from beta
        s4 = np.sum([(self.beta_shape_prior-1) * (digamma(self._beta_shape[d,k]) - np.log(self._beta_rate[d,k])) - self.beta_rate_prior * self._beta_shape[d,k]/ self._beta_rate[d,k] for d in range(self.D) for k in range(self.T)])
        
        # from s
        s5 = np.sum([ (self.alpha -1) * (digamma(self._s_shape[u])- np.log(self._s_rate[u])) - self.s_rate_prior * self._s_shape[u]/self._s_rate[u] for u in range(self.U)])
        
        # normalizer for s
        s6 = -np.sum([digamma(self._s_shape[u]) - np.log(self._s_rate[u]) for u in range(self.U)])

        # normalizer for beta
        s7 = -np.sum([digamma(self._beta_shape[d,k]) - np.log(self._beta_rate[d,k]) for d in range(self.D) for k in range(self.T)])
        
        print(s1,s2,s3,s4,s5,s6,s7)
        return s1 + s2 + s3 + s4 + s5 + s6 + s7 


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
    