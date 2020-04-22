import numpy as np
import keras.backend as K
import tensorflow as tf


def gmm_posterior_expected_value(components, z, noisesig):
    """
    Args:
        noisesig: float
        z: float
        components: np array, of the form:
            [ 3d mus matrix, 3d sigmas matrix, 3d weights matrix ]
    """
    sqr = K.square

    # constant factor
    const = K.exp( -sqr(z) / (2 * sqr(noisesig) ) )

    # numerator and denominator summations, for each distribution in components
    numerator = 0
    denominator = 0
    for i in range(len(components[0][0][0])):
        mu  = components[0,...,i]
        sig = components[1,...,i]
        wt  = components[2,...,i]

        num_term = wt * ( sqr(noisesig) * mu + sqr(sig) * z )
        num_term *= K.exp( -sqr(mu) / (2 * sqr(sig)) )
        num_term *= K.exp(
            ( sqr( sqr(noisesig) * mu + sqr(sig) * z ) ) / 
            ( 2 * sqr(noisesig) * sqr(sig) * (sqr(noisesig) + sqr(sig)) ) 
        )
        num_term /= K.pow( (sqr(noisesig) + sqr(sig)), 3/2 )
        numerator += num_term

        den_term = wt / (K.sqrt( sqr(noisesig) + sqr(sig) ))
        den_term *= K.exp(
            -(sqr(mu - z)) / 
            (2 * (sqr(noisesig) + sqr(sig)))
        )
        denominator += den_term
    
    return const * numerator / denominator



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