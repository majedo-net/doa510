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
        if self.N % 2:
            self.antennaPositions = np.arange(start=self.d*-(self.N-1)/2,stop=self.d*(self.N)/2,step=self.d)
        else:
            self.antennaPositions = np.arange(start=self.d*-(self.N-1)/2,stop=self.d*self.N/2,step=self.d)
        self.antennaPositions = np.vstack([self.antennaPositions,np.zeros_like(self.antennaPositions)])
        # weights used for hybrid beamforming, initialize as ones
        self.ws = np.eye(self.N)

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

    def hybridWeights(self,freq,theta0,phi0,Nsub,subtype='adj'):
        '''
        calculate hybrid beamforming weights and output applied to vk
        
        Nsub: number of subarrays
        subtype: adjacent or interleaved subarrays
        '''
        if self.N % 2:
            print('Hybrid beamforming requires even number of elements for equal sized subarrays')
            raise NotImplementedError
        wavelength = 3e8 / freq
        wavenumber = 2*np.pi/wavelength
        wavevector = -wavenumber*np.asarray([[np.sin(theta0)*np.cos(phi0)],
                                               [np.cos(theta0)*np.sin(phi0)]])
        wk = np.diag(np.exp(1j*wavevector.T@self.antennaPositions).squeeze())

        # subarray masking matrix for different subarray types
        match subtype:
            case 'adj':
                '''
                Construct as a block diagonal matrix:
                    P = 
                        [
                        A 0 ...
                        0 A ...
                        ... A_Nsub
                        ]
                    where A is (Nsub/Nantenna) x 1 vector of ones
                '''
                P = np.kron(np.eye(Nsub),np.ones([1,int(self.N / Nsub)]))
                self.ws = P @ wk

            case 'interleaved':
                '''
                
                '''
                P = np.zeros(self.N)
                P[::Nsub] = 1
                P = np.vstack([P, np.roll(P,1)])
                self.ws = P@wk




