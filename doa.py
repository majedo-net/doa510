import numpy as np

def w_mvdr(vk,sn):
    '''
    compute Minimum variance distortionless response weights for vk array manifold vector and sn received signal

    Van Trees presents this in frequency domain in section 6.2.1.1, copying PySDR time domain for now
    '''

    R = (sn @ sn.conj().T)/sn.shape[1]
    Rinv = np.linalg.pinv(R)
    w = (Rinv @ sn)/(sn.conj().T @ Rinv @ sn)
    return w

def power_mvdr(vk,sn):
    '''
    MVDR power spectrum trick from pysdr ebook
    '''
    R = np.cov(sn)
    Rinv = np.linalg.pinv(R)
    return (vk.conj().T @ Rinv @ vk).squeeze()