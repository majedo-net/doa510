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
    return 1/(vk.conj().T @ Rinv @ vk).squeeze()

def MUSIC(vk,sn,Ns):
    '''
    MUSIC beamformer
    vk: array manifold vector
    sn: received signal
    Ns: number of signals expected
    thetas: scan angles
    '''
    R = np.cov(sn)
    w, v = np.linalg.eig(R)
    eig_val_order = np.argsort(np.abs(w))
    v = v[:, eig_val_order]
    V = np.zeros((R.shape[0],R.shape[0]-Ns),dtype=np.complex64)
    for idx in range(R.shape[0]-Ns):
        V[:,idx] = v[:,idx]
    power_spectrum = []
    for idx in range(vk.shape[0]):
        vki = vk[idx,:].reshape(-1,1)
        metric = 1 / (vki.conj().T @ V @ V.conj().T @ vki)
        metric = np.abs(metric.squeeze())
        metric = 10*np.log10(metric)
        power_spectrum.append(metric)
    return power_spectrum

def DBT(rx,Pmax,d_over_lambda):
    '''
    Differential beam tracking
    rx: input signal Nsub x Nsample where rows are samples from each analog subarray of a hybrid beamformer
    Pmax: maximum power of the cross correlation, depends on Nsamples and antennas per subarray
    d_over_lambda: analog subarray effective spacing divided by wavelength
    '''
    Rx = np.correlate(rx[0,:],rx[1,:])/Pmax
    ux = -1*np.angle(Rx)
    uy = 0 # 1d case to start with
    doa_theta = np.sign(ux)*np.asin(np.sqrt(ux**2 + uy**2)/(2*np.pi*d_over_lambda))
    doa_phi = np.atan(uy/ux)
    return doa_theta.squeeze()
