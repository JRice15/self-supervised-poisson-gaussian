import numpy as np
import keras.backend as K
import tensorflow as tf


def make_gmm_pdf(components, locs, stds, weights):
    """
    returns callable pdf for these loc,std,weights
    """
    sum = 0
    for i in range(components):
    def pdf(x):
      y = weight / (std * np.sqrt(2 * np.pi))
      exp = -0.5 * ( (x - loc) / std )**2
      return y * np.exp(exp)
    return pdf

def gmm_prob(pdfs, x):
    sum = 0
    for p in pdfs:
        sum += pdf(x)
    return sum

def gmm_argmax(components, mus, vars, weights, z, noisevar):
    """
    Args:
        components: int, number of components
        mus: locs
        sigs: std devs
        weights: mixture weights / coefficients
        noisesig: float
        z: float
    """
     
    return result


def test_gm_expected():
    result = gmm_posterior_expected_value(
        np.array([ # 3 component mix
            [ # means
                [[-30, 70, 20],[400, 20, 20]],
                 [[50, -20, 20],[200, 0, 20]]
            ],[ # std devs
                [[100, 30, 30],[200, 12, 30]], 
                 [[20, 300, 30],[99, 2, 30]]
            ],[ # weights
                [[0.6, 0.4, 0],[0.2, 0.8, 0]],
                 [[0.5, 0.5, 0],[0.65, 0.35, 0]]
            ]
        ]), -710, 50
    )
    print(result)
    expected = -574
    assert (abs(result[0][0] - expected) < 0.001)
    print("success")
    


if __name__ == "__main__":
    test_gm_expected()
