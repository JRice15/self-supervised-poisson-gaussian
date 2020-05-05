import numpy as np
import keras.backend as K
import tensorflow as tf


def gmm_posterior_expected_value(components, mus, vars, weights, z, noisevar):
    """
    Args:
        components: int, number of components
        mus: locs
        sigs: std devs
        weights: mixture weights / coefficients
        noisesig: float
        z: float
    """
    z = np.squeeze(z.astype("float32"), axis=-1)
    zsq = z**2
    const = np.exp(-zsq/(2*noisevar))

    # numerator and denominator summations, for each distribution in components
    numerator = 0
    denominator = 0
    for i in range(components):
        # select each component layer
        mu  = mus[:,:,:,i]
        var = vars[:,:,:,i]
        wt  = weights[:,:,:,i]

        num_term = np.exp( (noisevar*(2*z-mu)*mu+zsq*var)/(2*noisevar*(noisevar+var)))
        num_term *= (noisevar*mu+z*var)
        num_term *= wt
        num_term /= np.power(noisevar+var,3/2)
        numerator += num_term

        den_term = np.exp(-((z-mu)**2)/(2*(noisevar+var)))
        den_term *= wt
        den_term /= np.sqrt(noisevar+var)
        denominator += den_term
    
    print('mus',np.any(np.isnan(mus)))
    print('vars',np.any(np.isnan(vars)))
    print('weights',np.any(np.isnan(weights)))
    print('const',np.any(np.isnan(const)))
    print('numerator',np.any(np.isnan(numerator)))
    print('denominator',np.any(np.isnan(denominator)))
    print(numerator.shape,denominator.shape,const.shape)
    result = const * numerator / (denominator+1e-10)
    print(result.shape)
     
    # replace nans with zero
    #result = tf.where(tf.math.is_nan(result), tf.fill(result.shape, 0.0), result)
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
