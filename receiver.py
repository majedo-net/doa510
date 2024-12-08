import numpy as np

class Receiver:
    '''
    Class to represent a receivers position, velocity, acceleration, and antenna

    '''
    def __init__(self,_x=[0.0,0.0,0.0],_v=[0.0,0.0,0.0],_a=[0.0,0.0,0.0]):
        self.x = np.asarray(_x) # initial position
        self.v = np.asarray(_v) # initial velocity
        self.a = np.asarray(_a) # initial acceleartion
        self.rx_signal = None

    def receiveSignal(self,transmitter, thetas, phis, channel,noise_type,**kwargs):
        '''
        receive a signal from the argument transmitter and the response channel

        need to have already calculated array manifold vector
        '''
        if "noise_power" in kwargs:
            self.pnoise = kwargs.pop('noise_power')
        # first find the angle of incidence
        theta_i = self.trueAoA(transmitter)
        theta_idx = (np.abs(thetas-theta_i)).argmin()
        if self.rx_signal is not None:
            # multiple transmitters
            self.rx_signal += self.ant.ws @ self.ant.vk[theta_idx,:].reshape(-1,1) @ transmitter.tx_signal.reshape(1,-1)
        else:
            self.rx_signal =  self.ant.ws @ self.ant.vk[theta_idx,:].reshape(-1,1) @ transmitter.tx_signal.reshape(1,-1)
        if channel:
            if noise_type==1:
                self.rx_signal += channel(self.pnoise,self.rx_signal.shape[1],self.rx_signal.shape[0])
            if noise_type==2:
                for i in range(self.rx_signal.shape[0]):
                    self.rx_signal[i,:] = self.rx_signal[i,:]*channel(100,10,self.rx_signal.shape[1])
            if noise_type==3:
                z,n = channel(self.pnoise,1000,100,self.rx_signal.shape[1],self.rx_signal.shape[0])
                for i in range(self.rx_signal.shape[0]):
                    self.rx_signal[i,:] = self.rx_signal[i,:]*z
                self.rx_signal += n

    
    def trueAoA(self,transmitter):
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
        return theta_i

