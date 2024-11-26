import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots
import log_setup as lgs

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
#plt.style.use(['science','ieee'])

class simulation_config:
    """
    Configuration file for a simulation. 
    Holds all relevant starting settings for a given simulation run.
    """
    def __init__(self, *args):
        if args != []:
            self.f0 = args[0]
            self.lambda0 = args[1]
            self.Fs = args[2]
            self.Nsamples = args[3]

            # vector of scan angles
            self.thetas = args[4]
            self.phis = args[5]

            # receiver
            self.rxer = args[6]
            self.rxer.ant = args[7]
            self.rxer.ant.computeVman(self.f0,self.thetas,self.phis)

            # transmitter
            self.theta0 = args[8]
            self.r0 = args[9]
            self.pos0 = args[10]
            self.txer = args[11]
            self.txer.ant = args[12]
            
            self.doa_method = args[13]

            if self.doa_method == "MUSIC":
                self.Ns = args[14]
            else:
                self.Ns = -1 # not using a method that requires this

        else:
            # Config is going to be set by access, not by a parameter list
            # Put placeholder generic arguments that may need to be corrected
            self.f0 = 0.5e9 # carrier frequency
            self.lambda0 = 3e8/self.f0
            self.Fs = 20e9 # oversample by a lot
            self.Nsamples = 10000

            # vector of scan angles
            self.thetas = np.linspace(-np.pi/2,np.pi/2,501)
            self.phis = np.zeros(1)

            # receiver with uniform linear array antenna
            self.rxer = receiver.Receiver()
            self.rxer.ant = antenna.Antenna('ula',d=self.lambda0/2,N=7)
            self.rxer.ant.computeVman(self.f0,self.thetas,self.phis)

            # transmitter at distance r0 and theta0 look angle with isotropic antenna
            self.theta0 = np.deg2rad(15)
            self.r0 = 20
            self.pos0 = self.r0*(np.asarray([np.cos(self.theta0),np.sin(self.theta0),0]))
            self.txer = transmitter.Transmitter(self.pos0)
            self.txer.ant = antenna.Antenna('iso')
            
            self.doa_method = "MUSIC"

            if self.doa_method == "MUSIC":
                self.Ns = 1
            else:
                self.Ns = -1 # not using a method that requires this

        # Acquire remaining derivable info from the info already in the config
        match self.doa_method:
            case "MVDR":
                
            case "MUSIC":
                power_spectrum = doa.MUSIC(self.rxer.ant.vk,self.rxer.rx_signal,Ns=1)
            case "ESPRIT":
                
            case _:
                raise Exception("Unknown DOA method")
        
def format_fig_to_file(plot_type, *args):
    match plot_type:
        case "rxed_signals":
            title_str, xlabel_str, ylabel_str, path_str = lgs.get_rxed_signals_plotting()
        
        case "estimate_AOA":
            title_str, xlabel_str, ylabel_str, path_str = lgs.get_estimate_AOA_plotting()
        
        case "custom":
            title_str = args[0]
            xlabel_str = args[0]
            ylabel_str = args[0]
            path_str = args[0]
            
        case _:
            raise Exception("Invalid plot_type")        
        
    plt.title(title_str)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(path_str)


def main(config):
    plt.figure()
    plt.plot(config.rxer.rx_signal[0,:].squeeze().real[0:200])
    plt.plot(config.rxer.rx_signal[1,:].squeeze().real[0:200])
    plt.plot(config.rxer.rx_signal[2,:].squeeze().real[0:200])
    format_fig_to_file("rxed_signals")

    plt.figure()
    plt.plot(np.rad2deg(config.thetas),power_spectrum)
    format_fig_to_file("estimate_AOA")

if __name__ == '__main__':
    simulator_settings = simulation_config()
    
    main(config = simulator_settings)