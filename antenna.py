import numpy as np

class Antenna:
    ''''
    Class to represent a transmitter/receiver objects antenna

    In the context of hybrid beamforming model, one antenna = one analog subarray

    For now, assume all array elements are at z=0 with respect to the transmitter/receiver position
    '''
    def __init__(self,_type='iso',**kwargs):
        match _type:
            case 'iso':
                self.initIso()
            case 'ula':
                try:
                    _d = kwargs.pop('d')
                    _N = kwargs.pop('N')
                    self.initULA(_d,_N)
                except KeyError:
                    print('provide element spacing d and number of element N for ULA')

    def initIso(self):
        '''
        Single isotropic antenna element
        '''
        # we can know the manifold vector for all frequencies right away for an isotropic element
        self.vk = 1
        # store the antenna positions just in case we try to compute vman 
        self.antennaPositions = np.zeros([1,1])


    def initULA(self,_d,_N):
        '''
        Uniform linear array with element spacing d (meters)
        By default the array will be created with elements spaced along the x-axis
        '''
        self.d = _d
        self.N = _N
        self.antennaPositions = np.arange(start=self.d*-self.N/2,stop=self.d*self.N/2,step=self.d)
        self.antennaPositions = np.vstack([self.antennaPositions,np.zeros_like(self.antennaPositions)])

    def computeVman(self,freq,thetas,phis):
        '''
        calculate the array manifold vector 
        Equation 2.28 in Van Trees
            what about (4.53) version? I think I have ended up with the same result here, double check
            ignoring z-axis for now

        freq: vector of frequencies in hertz
        thetas: vector of incident theta angles in radians
        phis: vector of incident phi angles in radians
        '''

        wavelengths = 3e8 / freq
        wavenumbers = 2*np.pi/wavelengths
        wavevectors = -wavenumbers*np.asarray([[np.sin(thetas)*np.cos(phis)],
                                               [np.cos(thetas)*np.sin(phis)]])
        self.vk = np.exp(-1j*wavevectors.T@self.antennaPositions)
        self.vk = np.squeeze(self.vk)



