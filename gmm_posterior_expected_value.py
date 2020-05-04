import numpy as np
import keras.backend as K
import tensorflow as tf

def p(i, v, i2=""):
    print(i, i2, K.eval(K.mean(v)))


def gmm_posterior_expected_value(components, mus, sigs, weights, z, noisesig):
    """
    Args:
        components: int, number of components
        mus: locs
        sigs: std devs
        weights: mixture weights / coefficients
        noisesig: float
        z: float
    """
    sqr = K.square
    z = K.squeeze(K.cast(z, "float32"), axis=-1)

    # constant factor
    const = K.exp( -sqr(z) / (2 * sqr(noisesig)) )

    # numerator and denominator summations, for each distribution in components
    numerator = 0
    denominator = 0
    for i in range(components):
        # select each component layer
        mu  = mus[:,:,:,i]
        sig = sigs[:,:,:,i]
        wt  = weights[:,:,:,i]

        num_term = wt * ( sqr(noisesig) * mu + sqr(sig) * z )
        num_term *= K.exp( -sqr(mu) / (2 * sqr(sig)) )
        num_term /= K.pow( (sqr(noisesig) + sqr(sig)), 3/2 )
        exponent = K.clip(
            ( sqr( sqr(noisesig) * mu + sqr(sig) * z ) ) / 
            ( 2 * sqr(noisesig) * sqr(sig) * (sqr(noisesig) + sqr(sig)) ), -70, 70)
        num_term *= K.exp(exponent)
        numerator += num_term

        den_term = wt / (K.sqrt( sqr(noisesig) + sqr(sig) ))
        den_term *= K.exp(
            -(sqr(mu - z)) / 
            (2 * (sqr(noisesig) + sqr(sig)))
        )
        denominator += den_term
    
    result = const * numerator / denominator
    # replace nans with large positive
    result = tf.where(tf.math.is_nan(result), tf.fill(result.shape, 1e25), result)
    return result


def test_gm_expected():
    result = gmm_posterior_expected_value(
        components=3, 
        mus=tf.constant([
            [[-30, 70, 20],[400, 20, 20]],
             [[50, -20, 20],[200, 0, 20]]
        ]),
        sigs=tf.constant([
            [[100, 30, 30],[200, 12, 30]], 
             [[20, 300, 30],[99, 2, 30]]
        ]),
        weights=tf.constant([
            [[0.6, 0.4, 0],[0.2, 0.8, 0]],
             [[0.5, 0.5, 0],[0.65, 0.35, 0]]
        ]),
        z=-710,
        noisesig=50
    )
    result = K.eval(result)
    expected = -574
    assert (abs(result[0][0] - expected) < 0.001)
    print("success")
    


if __name__ == "__main__":
    test_gm_expected()
