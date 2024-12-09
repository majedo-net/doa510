import numpy as np

class Transmitter:
    '''
    Class to represent a transmitters position, velocity, acceleration, antenna, and signal characteristics

    '''
    def __init__(self,_x=[0.0,0.0,0.0],_v=[0.0,0.0,0.0],_a=[0.0,0.0,0.0]):
        self.x = np.asarray(_x) # initial position
        self.v = np.asarray(_v) # initial velocity
        self.a = np.asarray(_a) # initial acceleartion

    def generateTone(self,freq,sample_rate,Nsamples):
        '''
        generate a single frequency sinusoid signal

        freq: carrier frequency in Hz
        sample_rate: in hertz
        Nsamples: number of samples to generate
        '''
        t = np.arange(Nsamples)/sample_rate
        self.tx_signal = np.exp(2j * np.pi*freq*t)
        return self.tx_signal

    def generateBPSK(self,freq,sample_rate,symb_rate,data):
        '''
        generate and upconvert a BPSK signal
        number of samples will be len(data)/symb_rate

        freq: carrier frequency in hertz
        sample_rate: in hertz
        symb_rate: symbol rate in hertz, i.e. signal bandwidth
        data: binary data stream of -1 and 1
        '''
        t = np.arange(0,len(data)/symb_rate,1/sample_rate)
        carrier = np.exp(2j*np.pi*freq*t)
        baseband =  np.repeat(data,int(sample_rate/symb_rate))
        self.tx_signal = carrier * baseband
        return self.tx_signal

    def timeStep(self,delta_t):
        self.v = self.v + delta_t * self.a
        self.x = self.x + delta_t * self.v
