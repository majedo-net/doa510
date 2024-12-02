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
    theta1 = np.deg2rad(-10)
    r0 = 20
    pos0 = r0*(np.asarray([np.cos(theta0),np.sin(theta0),0]))
    pos1 = r0*(np.asarray([np.cos(theta1),np.sin(theta1),0]))
    txer0 = transmitter.Transmitter(pos0)
    txer0.ant = antenna.Antenna('iso')
    txer1 = transmitter.Transmitter(pos1)
    txer1.ant = antenna.Antenna('iso')

    # generate the transmit signal
    txer0.generateTone(f0,Fs,Nsamples=Nsamples)
    txer1.generateTone(f0,Fs,Nsamples=Nsamples)

    # receive the signal(s)
    rxer.receiveSignal(txer0,thetas,phis,channel.awgn,noise_power=-15)
    rxer.receiveSignal(txer1,thetas,phis,channel.awgn,noise_power=-15)

    # check that signals are delayed
    # if you decrease theta0 the delay should get smaller
    # increase theta0 to 90 and delay should be half period
    plt.plot(rxer.rx_signal[0,:].squeeze().real[0:200])
    plt.plot(rxer.rx_signal[1,:].squeeze().real[0:200])
    plt.plot(rxer.rx_signal[2,:].squeeze().real[0:200])
    plt.show()

    # plotting manifold vector for sanity checks...
    #plt.plot(np.rad2deg(thetas),np.sum(rxer.ant.vk,1))
    #plt.show()


    # now lets estimate angle of arrival using MUSIC
    power_spectrum = doa.MUSIC(rxer.ant.vk,rxer.rx_signal,Ns=2)
    plt.plot(np.rad2deg(thetas),power_spectrum)
    plt.show()
    print(f'DOA Estimate = {np.rad2deg(thetas[np.argmax(power_spectrum)])}')