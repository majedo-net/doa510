import numpy as np

def awgn(noise_power,Nsamples):
    '''
    additive white gaussian noise channel
    noise_power in db
    Nsamples is number of samples 
    '''

    p = 10 ** (noise_power/10) # convert noise power to linear
    n = p*(np.random.randn(Nsamples) + 1j*np.random.randn(Nsamples))/np.sqrt(2)
    return n
