import numpy as np 
from numpy import random
import scipy.stats as stats
import os
import json
import scipy
import copy 
import scipy.sparse as sparse
import time

def read(folder):
    tags_file = 'tags.csv'
    links_file = 'links.csv'
    ratings_file = 'ratings.csv'
    movies_file = 'movie.csv'
    rating_train_file = 'rating_train.npz'
    rating_valid_file = 'rating_valid.npz'
    rating_test_file = 'rating_test.npz'
    map_file = 'movie_map.json'
    
    if os.path.exists(os.path.join(folder, rating_train_file)):
        rating_train = sparse.load_npz(os.path.join(folder, rating_train_file))
        rating_valid = sparse.load_npz(os.path.join(folder, rating_valid_file))
        rating_test = sparse.load_npz(os.path.join(folder, rating_test_file))
        with open(os.path.join(folder, map_file)) as file:
            movie_map = json.load(file)
    else:
        movie_map = {}
        i = 0 
        count = 0
        users,movies,ratings = [],[],[]
        with open(os.path.join(folder, ratings_file), 'r') as file:
            while True:
                line = file.readline()
                print(count)
                count +=1
                if not line:
                    break
                elif not line.startswith('user'):
                    try:
                        userId, movieId, rating, _ = line.split(',')
                        movieId, userId, rating =  int(movieId), int(userId), float(rating)
                        assert(rating in np.arange(0.5,5.5,0.5))
                        if movieId not in movie_map:
                            movie_map[movieId] = i
                            i += 1
                        ratings.append(rating * 2) #NOTE: Multiply by 2 for Poisson
                        users.append(userId-1)
                        movies.append(movie_map[movieId])
                    except:
                        pass
        #split into train, valid and test set
        users,movies,ratings = np.array(users),np.array(movies),np.array(ratings)
        split = random.choice(3, len(ratings),p=[0.75,0.05,0.2])
        rating_train = sparse.csr_matrix((ratings[split == 0],(users[split == 0], movies[split == 0])), shape = (610,i))
        sparse.save_npz(os.path.join(folder, rating_train_file),rating_train)
        rating_valid = sparse.csr_matrix((ratings[split == 1],(users[split == 1], movies[split == 1])), shape = (610,i))
        sparse.save_npz(os.path.join(folder, rating_valid_file), rating_valid)
        rating_test = sparse.csr_matrix((ratings[split == 2],(users[split == 2], movies[split == 2])), shape = (610,i))
        sparse.save_npz(os.path.join(folder, rating_test_file), rating_test)
        with open(os.path.join(folder, map_file), 'w') as file:
            json.dump(movie_map, file)
    return rating_train, rating_valid, rating_test, movie_map

def gibbs(rating_train, rating_valid, **kwargs):
    #Working with sparse matrix
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
    K = 10 #number of clusters
    a0,b0,a1 = 0.3,1,0.3  #global parameters for all users
    m0,n0,m1 = 0.3,1,0.3  #global parameters for all movies
    xi  = random.gamma(shape = a0, scale = 1/b0, size = U) #local latent per player
    eta = random.gamma(shape = m0, scale = 1/n0, size = D) #local latent per movie
    theta = np.array([random.gamma(shape = a1, scale = 1/xi[u], size = K) for u in range(U)]) #preferences per user
    beta = np.array([random.gamma(shape = m1, scale = 1/eta[d], size = K) for d in range(D)]) #attributes per movies
    assert theta.shape == (U,K)
    assert beta.shape == (D,K)
    
    z = np.zeros(shape = (U,D,K))
    for u,d in nonzero:
        z[u,d,:] = random.multinomial(rating_train[u,d], [1/K] * K)
    
    #Main algorithm
    max_iter = kwargs.pop('max_iter', 1000)
    threshold = kwargs.pop('threshold', 10e-4)
    logging = kwargs.pop('logging', True)
    n_iter = 0 
    sample = {'l':[], 'theta':[], 'z':[], 'beta':[], 'xi':[], 'eta':[]}
    
    def joint_log(rating, theta, beta, z, xi, eta):
        U,D = rating.shape
        return np.sum([stats.poisson.logpmf(rating[u,d],theta[u,:].T @ beta[d,:]) for u in range(U) for d in range(D)]) \
            + np.sum([stats.gamma.logpdf(theta[u,k],a = a1, scale = 1/xi[u]) for u in range(U) for k in range(K)]) \
            + np.sum([stats.gamma.logpdf(beta[d,k],a = m1, scale = 1/eta[d]) for d in range(D) for k in range(K)]) \
            + np.sum([stats.gamma.logpdf(xi[u],a = a0, scale = 1/b0) for u in range(U)]) \
            + np.sum([stats.gamma.logpdf(eta[d],a = m0, scale = 1/n0) for d in range(D)])

    while True:
        #Sample theta
        for u in range(U):
            for k in range(K):
                theta[u,k] = random.gamma(shape = a1 + np.sum(z[u,:,k]), scale = 1/(xi[u] + np.sum(beta[:,k])), size = 1)
        sample['theta'].append(theta)

        #Sample beta
        for d in range(D):
            for k in range(K):
                beta[d,k] = random.gamma(shape = m1 + np.sum(z[:,d,k]), scale = 1/(eta[d] + np.sum(theta[:,k])), size = 1)
        sample['beta'].append(beta)

        #Sample xi
        for u in range(U):
            xi[u] = random.gamma(shape = a0 + K * a1, scale = 1/(b0 + np.sum(theta[u,:])), size = 1)
        sample['xi'].append(xi)

        #Sample eta
        for d in range(D):
            eta[d] = random.gamma(shape = m0 + K * m1, scale = 1/(n0 + np.sum(beta[d,:])), size = 1)
        sample['eta'].append(eta)

        #Sample z
        for u,d in nonzero:
            z[u,d,:] = random.multinomial(rating_train[u,d], theta[u,:].T * beta[d,:]/(theta[u,:].T @ beta[d,:]))
        sample['z'].append(z)
        
        
        #Validate
        training_likelihood = validate(theta, beta, rating_train)
        val_likelihood = validate(theta, beta, rating_valid)
        print(f'Iter {n_iter}: training_error = {training_likelihood}, val_error = {val_likelihood}, max_pref = {np.sum(theta,0).max()}, min_pref = {np.sum(theta,0).min()},max_att = {np.sum(beta,0).max()}, min_att = {np.sum(beta,0).min()}')
        # if n_iter > 0:
        #     if abs(val_likelihood/last_val_likelihood-1) < threshold or n_iter >= max_iter:
        #         print(f'Compete after {n_iter} iterations: Validation Error = {val_likelihood}')
        #         break
        last_val_likelihood = val_likelihood
        n_iter += 1 
    return sample 

def mixed_membership():
    pass

def validate(theta,beta,rating_valid):
    size = len(list(zip(*rating_valid.nonzero())))
    rating_valid = rating_valid.toarray().flatten()
    mu = (theta @ beta.T).flatten()
    mu = mu[rating_valid > 0]
    rating_valid = rating_valid[rating_valid > 0]
    mu[mu>10] = 10
    assert mu.shape == rating_valid.shape, f'{mu.shape} vs {rating_valid.shape}'
    return (np.sum(rating_valid * np.log(mu)) - np.sum(mu))/size
    #nonzero = list(zip(*rating_valid.nonzero()))
    #return np.sum([rating_valid[u,d] * np.log(theta[u,:] @ beta[d,:].T) for u,d in nonzero]) - np.sum(theta,0) @ np.sum(beta,0)
    

def vi(rating_train, rating_valid, **kwargs):
    #Working with sparse matrix
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
    K = 10 #number of clusters
    a0,b0,a1 = 0.3,1,0.3  #global parameters for all users
    m0,n0,m1 = 0.3,1,0.3  #global parameters for all movies
    #xi  = random.gamma(shape = a0, scale = 1/b0, size = U) #local latent per player
    #eta = random.gamma(shape = m0, scale = 1/n0, size = D) #local latent per movie
    #theta = np.array([random.gamma(shape = a1, scale = 1/xi[u], size = K) for u in range(U)]) #preferences per user
    #beta = np.array([random.gamma(shape = m1, scale = 1/eta[d], size = K) for d in range(D)]) #attributes per movies
    #assert theta.shape == (U,K)
    #assert beta.shape == (D,K)

    #z = [[random.multinomial(rating[u,indices[d]], [1/K] * K) for d,u in range(indptr[u], indptr[u+1])] for u in range(U)]
    #assert z.shape == (U,D,K)
       
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

if __name__ == "__main__":
    rating_train, rating_valid, rating_test, movie_map = read()
    #rating = np.array([[random.poisson(3) for i in range(2000)] for j in range(100)])
    gibbs(rating_valid, rating_valid,max_iter = 50 )
    
    # model = vi(rating_valid, rating_valid, max_iter = 50)