import numpy as np


def gmm_posterior_expected_value(components, z, noisesig):
    """
    Args:
        noisesig: float
        z: float
        components: np array, of the form:
            [ 3d mus array, 3d sigmas array, 3d weights array ]
    """
    # constant factor
    const = np.exp( -(z**2) / (2 * noisesig**2) )

    # numerator and denominator summations, for each distribution in components
    numerator = 0
    denominator = 0
    for i in range(len(components[0][0][0])):
        mu  = components[0,...,i]
        sig = components[1,...,i]
        wt  = components[2,...,i]
        print(i, mu)
        print(sig)
        print(wt)

        num_term = wt * ( (noisesig**2) * mu + (sig**2) * z )
        num_term *= np.exp( -(mu**2) / (2 * sig**2) )
        num_term *= np.exp(
            ( ( (noisesig**2) * mu + (sig**2) * z )**2 ) / 
            ( 2 * (noisesig**2) * (sig**2) * (noisesig**2 + sig**2) ) 
        )
        num_term /= (noisesig**2 + sig**2)**(3/2)
        numerator += num_term

        den_term = wt / (np.sqrt( noisesig**2 + sig**2 ))
        den_term *= np.exp(
            -((mu - z)**2) / 
            (2 * (noisesig**2 + sig**2))
        )
        denominator += den_term
    
    return const * numerator / denominator



def test_gm_expected():
    result = gaussian_mixture_expected_value(
        50, -710, 
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
        ])
    )
    print(result)
    expected = -574
    assert (abs(result[0][0] - expected) < 0.001)
    print("success")
    


if __name__ == "__main__":
    test_gm_expected()