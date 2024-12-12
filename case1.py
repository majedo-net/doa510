#%%
import time
import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = True

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
    thetas = np.linspace(-np.pi/2,np.pi/2,51)
    phis = np.zeros(1)

    
    '''
    Set up the receivers and transmitters
    ======================================================================================
    '''

    # digital receiver can be shared for both digital methods
    dig_rxer = receiver.Receiver()
    dig_rxer.ant = antenna.Antenna('ula',d=dspace,N=Nant)
    dig_rxer.ant.computeVman(f0,thetas,phis)
    # analog receiver needs to be separate due to hybrid weights
    ana_rxer = receiver.Receiver()
    ana_rxer.ant = antenna.Antenna('ula',d=dspace,N=Nant)
    ana_rxer.ant.computeVman(f0,thetas,phis)

    # initial guess for transmitter direction is theta = 0
    th_guess = 60

    # transmitter at distance r0 and theta0 starting position look angle with isotropic antenna
    theta0 = np.deg2rad(60)
    r0 = 20 # meters
    pos0 = r0*(np.array([np.cos(theta0),np.sin(theta0),0]))
    vel0 = np.array([0,-0.2,0])
    txer = transmitter.Transmitter(pos0,vel0)
    txer.ant = antenna.Antenna('iso')

    # generate the transmit signal
    #txer.generateTone(f0,Fs,Nsamples=Nsamples)
    data = 2* np.random.randint(0,2,32) -1
    txer.generateBPSK(f0,Fs,100e6,data)

    # generate initial estimate for aoa
    ana_rxer.ant.hybridWeights(f0,th_guess,0,Nsub=Nsubarrays,subtype='interleaved')
    ana_rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=5)

    doa_error = doa.DBT(ana_rxer.rx_signal,Pmax,dspace/lambda0)

    '''
    Run the simulation scene Nrepeat times to generate statistics
    ======================================================================
    '''
    Nrepeat = 10
    # begin time stepping through epochs
    Nepochs = 200
    
    mvdr_rmse = []
    music_rmse = []
    dbt_rmse = []
    mvdr_var = []
    music_var = []
    dbt_var = []
    
    noise_powers = [-10,-5,0,5,10,15,20]
    
    for npidx in noise_powers:
        print(f'Starting noise power {npidx}')
        dbt_errors = []
        music_errors = []
        mvdr_errors = []
        for nidx in range(Nrepeat):
            xs = []
            ys = []
            mvdr_est_doa = []
            music_est_doa = []
            dbt_est_doa = []
            true_doa = []
            th_guess = 60
            for ti in range(Nepochs):
                txer.timeStep(1)

                # DBT th_guess wrapping 
                th_guess = th_guess + doa_error
                if th_guess > np.pi/2:
                    th_guess =- np.pi/2
                elif th_guess < -np.pi/2:
                    th_guess =+np.pi/2

                # update hybrid weights and receive the signal
                ana_rxer.ant.hybridWeights(f0,th_guess,0,Nsub=Nsubarrays,subtype='interleaved')
                ana_rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=npidx)
                dig_rxer.receiveSignal(txer,thetas,phis,channel.awgn,noise_power=npidx)
                true_doa.append(np.rad2deg(ana_rxer.trueAoA(txer)))
                

                # differential beam tracking (DBT)
                doa_error =  doa.DBT(ana_rxer.rx_signal,Pmax,dspace/lambda0)
                dbt_est_doa.append(np.rad2deg(th_guess + doa_error))
                
                # MVDR
                power_spectrum = []
                for thidx in range(thetas.size):
                    power_spectrum.append(doa.power_mvdr(dig_rxer.ant.vk[thidx,:],dig_rxer.rx_signal))
                power_spectrum = 10*np.log10(np.abs(power_spectrum))
                mvdr_est_theta = np.rad2deg(thetas[np.argmax(power_spectrum)])
                mvdr_est_doa.append(mvdr_est_theta)

                # MUSIC
                power_spectrum = []
                power_spectrum = doa.MUSIC(dig_rxer.ant.vk,dig_rxer.rx_signal,Ns=1)
                power_spectrum = 10*np.log10(np.abs(power_spectrum))
                music_est_theta = np.rad2deg(thetas[np.argmax(power_spectrum)])
                music_est_doa.append(music_est_theta)

                xs.append(txer.x[0])
                ys.append(txer.x[1])

                # clear the received signals
                ana_rxer.rx_signal = None
                dig_rxer.rx_signal = None
                # reset the transmitter position

                # end time for
            txer.x = pos0
            dbt_errors.append(np.sum((np.asarray(dbt_est_doa)-np.asarray(true_doa))**2))
            music_errors.append(np.sum((np.asarray(mvdr_est_doa)-np.asarray(true_doa))**2))
            mvdr_errors.append(np.sum((np.asarray(mvdr_est_doa)-np.asarray(true_doa))**2))
            plt.figure()
            plt.plot(np.arange(Nepochs),dbt_est_doa,label='DBT Estimated AoA')
            plt.plot(np.arange(Nepochs),mvdr_est_doa,label='MVDR Estimated AoA')
            plt.plot(np.arange(Nepochs),music_est_doa,label='MUSIC Estimated AoA')
            plt.plot(np.arange(Nepochs),true_doa,label='True AoA')
            plt.xlabel('Time Snapshot')
            plt.ylabel(r'Angle of arrival ($\theta$)')
            plt.legend()
            plt.savefig(f'report_figures/case1/snr_{npidx*-1}_iter_{nidx}.png')
            print(f'Iteration {nidx} finished')
            #end monte carlo for
        dbt_rmse.append(np.sqrt(np.sum(dbt_errors)/(Nrepeat*Nepochs)))
        mvdr_rmse.append(np.sqrt(np.sum(mvdr_errors)/(Nrepeat*Nepochs)))
        music_rmse.append(np.sqrt(np.sum(music_errors)/(Nrepeat*Nepochs)))

        dbt_var.append(np.var(np.asarray(dbt_errors)/Nepochs))
        mvdr_var.append(np.var(np.asarray(mvdr_errors)/Nepochs))
        music_var.append(np.var(np.asarray(music_errors)/Nepochs))
        #end noise power for
            
        
    plt.figure()
    plt.plot(-1*np.asarray(noise_powers),10*np.log10(dbt_rmse),label='DBT')
    plt.plot(-1*np.asarray(noise_powers),10*np.log10(mvdr_rmse),label='MVDR')
    plt.plot(-1*np.asarray(noise_powers),10*np.log10(music_rmse),label='MUSIC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (db(degrees))')
    plt.legend()
    plt.savefig('report_figures/case1.png')
    