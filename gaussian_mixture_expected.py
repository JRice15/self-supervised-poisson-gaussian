import numpy as np


def gaussian_mixture_expected_value(noisesig, z, components):
    """
    Args:
        noisesig: float
        z: float
        components: np array, of the form:
            [ [mu1, sigma1, weight1], ...]
    """
    # constant factor
    const = np.exp( -(z**2) / (2 * noisesig**2) )

    # numerator and denominator summations, for each distribution in components
    numerator = 0
    denominator = 0
    for i in range(len(components[0])):
        mu  = components[0,i]
        sig = components[1,i]
        wt  = components[2,i]
        print(mu, sig, wt)

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
        np.array([
            [[-30, 70]], 
            [[100, 30]], 
            [[0.6, 0.4]]
        ])
    )
    expected = -574
    print(result)
    assert (result - expected < 1e-5)

    print("success")
    


if __name__ == "__main__":
    test_gm_expected()