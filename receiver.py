import numpy as np

class Receiver:
    '''
    Class to represent a receivers position, velocity, acceleration, and antenna

    '''
    def __init__(self,_x=[0.0,0.0,0.0],_v=[0.0,0.0,0.0],_a=[0.0,0.0,0.0]):
        self.x = np.asarray(_x) # initial position
        self.v = np.asarray(_v) # initial velocity
        self.a = np.asarray(_a) # initial acceleartion

    def receiveSignal(self,transmitter, thetas, phis, channel):
        '''
        receive a signal from the argument transmitter and the response channel

        need to have already calculated array manifold vector
        '''
        # first find the angle of incidence
        # normalize both position vectors
        if np.linalg.norm(self.x) != 0:
            v1 = self.x / self.linalg.norm(self.x)
        else:
            v1 = self.x
        
        if np.linalg.norm(transmitter.x) != 0:
            v2 = transmitter.x / np.linalg.norm(transmitter.x)
        else:
            v2 = transmitter.x

        vd = v2 - v1
        vd = vd[0] + 1j*vd[1]
        theta_i = np.angle(vd)
        theta_idx = (np.abs(thetas-theta_i)).argmin()
        self.rx_signal = self.ant.vk[theta_idx,:].reshape(-1,1) @ transmitter.tx_signal.reshape(1,-1)
