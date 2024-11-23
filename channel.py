import numpy as np

def awgn(noise_power,Nsamples,Nrows):
    '''
    additive white gaussian noise channel
    noise_power in db
    Nsamples is number of samples 
    '''

    p = 10 ** (noise_power/10) # convert noise power to linear
    n = np.zeros([Nrows,Nsamples])
    for nr in range(Nrows):
        n[nr,:] = p*(np.random.randn(Nsamples) + 1j*np.random.randn(Nsamples))/np.sqrt(2)
    return n
