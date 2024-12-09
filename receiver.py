import numpy as np

class Receiver:
    '''
    Class to represent a receiver's position, velocity, acceleration, and antenna
    '''
    def __init__(self, _x=[0.0, 0.0, 0.0], _v=[0.0, 0.0, 0.0], _a=[0.0, 0.0, 0.0], noise_power=0):
        self.x = np.asarray(_x)  # initial position
        self.v = np.asarray(_v)  # initial velocity
        self.a = np.asarray(_a)  # initial acceleration
        self.rx_signal = None
        self.pnoise = noise_power  # Initialize noise power attribute

    def receiveSignal(self, transmitter, thetas, phis, channel, **kwargs):
        '''
        Receive a signal from the argument transmitter and the response channel
        '''
        # Update noise power if provided
        if "noise_power" in kwargs:
            self.pnoise = kwargs.pop('noise_power')

        # Find the angle of incidence
        theta_i = self.trueAoA(transmitter)
        theta_idx = (np.abs(thetas - theta_i)).argmin()

        # debug statement 
        # print(f"Receiving signal: theta_idx = {theta_idx}, theta_i = {theta_i}")
        
        # Handle isotropic and ULA antennas
        if len(self.ant.vk.shape) == 1:  # Handle isotropic antenna (1D vk)
            vk = self.ant.vk.reshape(-1, 1)
        else:
            vk = self.ant.vk[theta_idx, :].reshape(-1, 1)  # ULA or other types

        tx_signal = transmitter.tx_signal.reshape(1, -1)  # Ensure tx_signal is row vector
        
        # Accumulate received signals
        if self.rx_signal is not None:

           # print(f"[DEBUG receiveSignal] Before rx_signal update:")
           # print(f"  Shape of self.rx_signal: {self.rx_signal.shape if self.rx_signal is not None else 'None'}")
            # print(f"  Shape of self.ant.ws: {self.ant.ws.shape}")
           #  print(f"  Shape of self.ant.vk[theta_idx, :]: {self.ant.vk[theta_idx, :].shape}")
           #  print(f"  Shape of transmitter.tx_signal: {transmitter.tx_signal.shape}")


            self.rx_signal += (
                self.ant.ws
                @ self.ant.vk[theta_idx, :].reshape(-1, 1)
                @ transmitter.tx_signal.reshape(1, -1)
            )
           #  print(f"[DEBUG receiveSignal] After rx_signal update:")
           #  print(f"  Shape of self.rx_signal: {self.rx_signal.shape}")

        else:
            self.rx_signal = (
                self.ant.ws
                @ self.ant.vk[theta_idx, :].reshape(-1, 1)
                @ transmitter.tx_signal.reshape(1, -1)
            )

    # debug statement 
        # print(f"True AoA: {theta_i}, Index: {theta_idx}")

        # Add channel noise if applicable
        if channel:
            self.rx_signal += channel(self.pnoise, self.rx_signal.shape[1], self.rx_signal.shape[0])

    def trueAoA(self, transmitter):
        '''
        Compute the true angle of arrival based on transmitter position
        '''
        # Normalize position vectors
        if np.linalg.norm(self.x) != 0:
            v1 = self.x / np.linalg.norm(self.x)
        else:
            v1 = self.x

        if np.linalg.norm(transmitter.x) != 0:
            v2 = transmitter.x / np.linalg.norm(transmitter.x)
        else:
            v2 = transmitter.x

        vd = v2 - v1
        vd = vd[0] + 1j * vd[1]
        theta_i = np.angle(vd)
        return theta_i
