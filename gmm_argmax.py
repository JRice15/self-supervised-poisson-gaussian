import numpy as np
import keras.backend as K
import tensorflow as tf


def make_gmm_pdf(locs, stds, weights):
    """
    returns callable pdf for these loc,std,weights
    """
    def pdf(x):
        y = weights / (stds * np.sqrt(2 * np.pi))
        exp = -0.5 * ( (x - locs) / stds )**2
        y *= np.exp(exp)
        return np.sum(y, axis=-1)
    return pdf


def gmm_argmax(mus, std, weights):
    """
    """
    gmm_pdf = make_gmm_pdf(mus, std, weights)
    
    bestval = gmm_pdf(0)
    bestarg = np.zeros(bestval.shape)
    for i in np.linspace(0,1,256):
        if i == 0: continue
        newval = gmm_pdf(i)
        bestarg = np.where(newval > bestval, i, bestarg)
        bestval = np.where(newval > bestval, newval, bestval)
        
    return bestarg



def test_argmax():
    a = gmm_argmax(
                   np.array([[[2,-3],[4,-7]],[[-1,0.8],[-5,4]]]),
                   np.array([[[4,3],[6,5]],[[3,0.5],[2,4]]]),
                   np.array([[[.5,.5],[.5,.5]],[[.5,.5],[.5,.5]]])
                  )
    print(a)




if __name__ == "__main__":
    test_argmax()
