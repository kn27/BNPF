#import nonparametric
import scipy
from nonparametric import NPNMF, simulate
if __name__ == "__main__":
    X, theta, beta, s, v = simulate(U = 100,
    D = 30,
    K = 10,
    alpha=1.1,
    beta_shape_prior=0.3,
    beta_rate_prior=0.3,
    s_rate_prior=1)
    X = scipy.sparse.csr_matrix(X)
    nmf = NPNMF(X,T = 15, seed = 1, threshold = 1e-5)
    #nmf.load_model('model_1.npz')
    nmf.inference()
    # temp = copy.deepcopy(nmf._phi)
    # nmf = NPNMF(X,T = 30, seed = 0, threshold = 1e-8)
    #nmf.update_phi()
    # temp2 = copy.deepcopy(nmf._phi)
    print()
    