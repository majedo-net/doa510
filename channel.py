import numpy as np

def awgn(noise_power,Nsamples,Nrows):
    '''
    additive white gaussian noise channel
    noise_power in db
    Nsamples is number of samples 
    '''

    p = 10 ** (noise_power/10) # convert noise power to linear
    n = np.zeros([Nrows,Nsamples],dtype=np.complex64)
    for nr in range(Nrows):
        n[nr,:] = p*(np.random.randn(Nsamples) + 1j*np.random.randn(Nsamples))/np.sqrt(2)
    return n



def rlhf(tx_spd,num_refl,Nsamples):
    '''
    Rayleigh Fading channel
    This can be coupled with awgn noise
    This simulates the reflections of multiple sinusoids
    All rows have the same effect applied
    Multiply by z
    '''

    # Simulation Params, change these based on the simulation
    v_mph = tx_spd # velocity of either TX or RX, in miles per hour (typically 60)
    center_freq = 0.5e6 # RF carrier frequency in Hz
    Fs = 20e9 # sample rate of simulation
    N = num_refl # number of sinusoids to sum (typically 100)

    v = v_mph * 0.44704 # convert to m/s
    fd = v*center_freq/3e8 # max Doppler shift
    t = np.arange(0, Nsamples/Fs, 1/Fs) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

    # z is the complex coefficient representing channel, you can think of this as a phase shift and magnitude scale
    z = (1/np.sqrt(N)) * (x + 1j*y) # this is what you would actually use when simulating the channel
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag) # convert to dB

    return z

def both(noise_power,tx_spd,num_refl,Nsamples,Nrows):
    '''
    Calculate both noises and combine
    '''

    # Simulation Params, change these based on the simulation
    v_mph = tx_spd # velocity of either TX or RX, in miles per hour (typically 60)
    center_freq = 0.5e9 # RF carrier frequency in Hz
    Fs = 20e9 # sample rate of simulation
    N = num_refl # number of sinusoids to sum (typically 100)

    v = v_mph * 0.44704 # convert to m/s
    fd = v*center_freq/3e8 # max Doppler shift
    t = np.arange(0, Nsamples/Fs, 1/Fs) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

    # z is the complex coefficient representing channel, you can think of this as a phase shift and magnitude scale
    z = (1/np.sqrt(N)) * (x + 1j*y) # this is what you would actually use when simulating the channel
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag) # convert to dB

    p = 10 ** (noise_power/10) # convert noise power to linear
    n = np.zeros([Nrows,Nsamples],dtype=np.complex64)
    for nr in range(Nrows):
        n[nr,:] = p*(np.random.randn(Nsamples) + 1j*np.random.randn(Nsamples))/np.sqrt(2)

    return z,n
    
