import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
#plt.style.use(['science','ieee'])

if __name__ == '__main__':
    f0 = 0.5e9 # carrier frequency
    lambda0 = 3e8/f0
    Fs = 20e9 # oversample by a lot
    Nsamples = 10000

    # vector of scan angles
    thetas = np.linspace(-np.pi/2,np.pi/2,501)
    phis = np.zeros(1)

    # receiver with uniform linear array antenna
    rxer = receiver.Receiver()
    rxer.ant = antenna.Antenna('ula',d=lambda0/2,N=7)
    rxer.ant.computeVman(f0,thetas,phis)

    # transmitter at distance r0 and theta0 look angle with isotropic antenna
    theta0 = np.deg2rad(45)
    r0 = 20
    pos0 = r0*(np.asarray([np.cos(theta0),np.sin(theta0),0]))
    txer = transmitter.Transmitter(pos0)
    txer.ant = antenna.Antenna('iso')

    # generate the transmit signal
    txer.generateTone(f0,Fs,Nsamples=Nsamples)

    # receive the signal
    rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=14)

    # check that signals are delayed
    # if you decrease theta0 the delay should get smaller
    # increase theta0 to 90 and delay should be half period
    plt.figure()
    plt.plot(rxer.rx_signal[0,:].squeeze().real[0:200])
    plt.plot(rxer.rx_signal[1,:].squeeze().real[0:200])
    plt.plot(rxer.rx_signal[2,:].squeeze().real[0:200])
    plt.show()

    # plotting manifold vector for sanity checks...
    plt.plot(np.rad2deg(thetas),np.sum(rxer.ant.vk,1))
    plt.show()


    # now lets estimate angle of arrival using MUSIC
    power_spectrum = doa.MUSIC(rxer.ant.vk,rxer.rx_signal,Ns=1)
    plt.figure()
    plt.plot(np.rad2deg(thetas),power_spectrum)
    plt.show()
    print(f'DOA Estimate = {np.rad2deg(thetas[np.argmax(power_spectrum)])}')