import time
import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
#plt.style.use(['science','ieee'])

if __name__ == '__main__':
    f0 = 500e6 # carrier frequency
    lambda0 = 3e8/f0
    dspace = lambda0/2
    Fs = 10e9 # oversample by a lot
    Nsamples = 10000

    Nant = 16 # total number of antennas across all subarrays
    Nsubarrays = 2 # analog subarrays
    Pmax = (Nant/Nsubarrays)**2 * Nsamples # maximum power for the cross correlation
    deff = (Nant/Nsubarrays)*dspace # effective spacing of subarrays

    # vector of scan angles
    thetas = np.linspace(-np.pi/2,np.pi/2,501)
    phis = np.zeros(1)

    # receiver with uniform linear array antenna
    rxer = receiver.Receiver()
    rxer.ant = antenna.Antenna('ula',d=dspace,N=Nant)
    rxer.ant.computeVman(f0,thetas,phis)
    # initial guess for transmitter direction is theta = 0
    th_guess = 0

    # transmitter at distance r0 and theta0 look angle with isotropic antenna
    theta0 = np.deg2rad(60)
    r0 = 20
    pos0 = r0*(np.asarray([np.cos(theta0),np.sin(theta0),0]))
    vel0 = np.asarray([0,-0.2,0])
    txer = transmitter.Transmitter(pos0,vel0)
    txer.ant = antenna.Antenna('iso')

    # generate the transmit signal
    #txer.generateTone(f0,Fs,Nsamples=Nsamples)
    data = 2* np.random.randint(0,2,32) -1
    txer.generateBPSK(f0,Fs,100e6,data)

    # generate initial estimate for aoa
    rxer.ant.hybridWeights(f0,th_guess,0,Nsub=Nsubarrays,subtype='interleaved')
    rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=10)

    #plt.plot(rxer.rx_signal.squeeze().real[0,0:500])
    #plt.show()

    doa_error = doa.DBT(rxer.rx_signal,Pmax,dspace/lambda0)

    # begin time stepping through epochs
    Nepochs = 200
    est_doa = []
    true_doa = []
    xs = []
    ys = []
    for ti in range(Nepochs):
        txer.timeStep(1)

        th_guess = th_guess + doa_error
        if th_guess > np.pi:
            th_guess =- np.pi
        elif th_guess < -np.pi:
            th_guess =+np.pi
        # update hybrid weights and receive the signal
        rxer.ant.hybridWeights(f0,th_guess,0,Nsub=Nsubarrays,subtype='interleaved')
        rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=0)

        # differential beam tracking (DBT)
        doa_error =  doa.DBT(rxer.rx_signal,Pmax,dspace/lambda0)
        est_doa.append(np.rad2deg(th_guess + doa_error))
        true_doa.append(np.rad2deg(rxer.trueAoA(txer)))
        xs.append(txer.x[0])
        ys.append(txer.x[1])
        

    plt.scatter(xs,ys)
    plt.scatter(rxer.ant.antennaPositions[1,:],rxer.ant.antennaPositions[0,:])
    plt.show()

    plt.plot(np.arange(Nepochs),est_doa)
    plt.plot(np.arange(Nepochs),true_doa)
    plt.show()
    