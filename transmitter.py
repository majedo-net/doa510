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
        t = np.arange(Nsamples)/sample_rate
        self.tx_signal = np.exp(2j * np.pi*freq*t)
        return self.tx_signal

    def timeStep(self,delta_t):
        self.v = delta_t * self.a
        self.x = delta_t * self.v
