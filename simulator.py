import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots
import logging
import os
import functools
from datetime import datetime

logger = logging.getLogger(__name__)

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
plt.style.use(['science','ieee'])

# Simulation config class to bundle all the associated simulation data together
class simulation_config:
    """
    Configuration file for a simulation. 
    Holds all relevant starting settings for a given simulation run.
    """
    def __init__(self, *args):
        if args != []:
            options = args[0]
            
            self.f0 = options[0]
            self.lambda0 = 3e8 / self.f0
            self.Fs = options[1] * self.f0
            self.Nsamples = options[2]

            # vector of scan angles
            self.thetas = options[3]
            self.phis = options[4]

            # receiver
            self.rxer_description = options[5]
            match self.rxer_description:
                case "":
                    self.rxer = receiver.Receiver()
                case _:
                    raise Exception("transmitter type not implemented")
            
            self.rxer_ant_type = options[6]
            self.rxer_ant_num = options[7]
            if self.rxer_ant_type == 'iso':
                self.rxer.ant = antenna.Antenna('iso')
            else:
                self.rxer.ant = antenna.Antenna(self.rxer_ant_type,d=self.lambda0/2,N=self.rxer_ant_num)
            
            self.rxer.ant.computeVman(self.f0,self.thetas,self.phis)

            # signal arrival
            self.theta0 = options[8]
            self.r0 = options[9]
            self.pos0 = self.r0*(np.asarray([np.cos(self.theta0),np.sin(self.theta0),0]))
            
            # transmitter
            self.txer_description = options[10]
            match self.txer_description:
                case "":
                    self.txer = transmitter.Transmitter(self.pos0)
                case _:
                    raise Exception("transmitter type not implemented")
            
            self.txer_ant_type = options[11]
            self.txer_ant_num = options[12]
            if self.txer_ant_type == 'iso':
                self.txer.ant = antenna.Antenna('iso')
            else:
                self.txer.ant = antenna.Antenna(self.txer_ant_type,d=self.lambda0/2,N=self.txer_ant_num)

            self.doa_method = options[13]

            if self.doa_method == "MUSIC":
                self.Ns = options[14]
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
        # generate the transmit signal
        self.txer.generateTone(self.f0,self.Fs,Nsamples=self.Nsamples)

        # receive the signal
        self.rxer.receiveSignal(self.txer,self.thetas,self.phis,channel.awgn)        

        match self.doa_method:
            case "MVDR":
                self.power_spectrum = []
                for thi in range(self.thetas.size):
                    self.power_spectrum.append(doa.power_mvdr(self.rxer.ant.vk[thi,:],self.rxer.rx_signal))
                self.power_spectrum = 10*np.log10(np.real(self.power_spectrum)) # convert to dB
                
            case "MUSIC":
                self.power_spectrum = doa.MUSIC(self.rxer.ant.vk,self.rxer.rx_signal,Ns=1)

            case _:
                raise Exception("Unknown DOA method")


# Plotting utility functions
def get_rxed_signals_plotting(fig_path):
    title_str = "Received Signals Sampled"
    xlabel_str = "Sample Number"
    ylabel_str = "Magnitude"
    path_str = fig_path + "_rxed_signals.png"

    plot_info_tuple = (title_str, xlabel_str, ylabel_str, path_str)

    return plot_info_tuple

def get_estimate_AOA_plotting(fig_path, AOA_method):
    title_str = "Estimated Angle of Arrival Using " + AOA_method
    xlabel_str = "Angle (Degrees)"
    ylabel_str = "AOA Metric"
    path_str = fig_path + "_rxed_signals.png"

    plot_info_tuple = (title_str, xlabel_str, ylabel_str, path_str)

    return plot_info_tuple   

def format_fig_to_file(plot_type, fig_trail, AOA_estimator = "not_used", *args):
    match plot_type:
        case "rxed_signals":
            title_str, xlabel_str, ylabel_str, path_str = get_rxed_signals_plotting(fig_trail)
        
        case "estimate_AOA":
            title_str, xlabel_str, ylabel_str, path_str = get_estimate_AOA_plotting(fig_trail, AOA_estimator)
        
        case "custom":
            title_str = args[0]
            xlabel_str = args[1]
            ylabel_str = args[2]
            path_str = "fig_path" + args[3]
            
        case _:
            raise Exception("Invalid plot_type")        
        
    plt.title(title_str)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(path_str)


# Logging functions
def log_config_simulation(config, log_path, tree_code):
    logging.basicConfig(filename=log_path, level=logging.INFO)
    
    now = datetime.now()
    logger.info(f"Simulation run started logging at: {now}")

    # Log the parameters of the configuration
    logger.info(f"Configutation parameters: ")
    logger.info(f"f0: {config.f0}")
    logger.info(f"Fs / f0 ratio: {config.Fs / config.f0}")
    logger.info(f"Nsamples: {config.Nsamples}")
    
    # logger.info(f"thetas: {config.thetas}")
    # logger.info(f"phis: {config.phis}")
    
    logger.info(f"Receiver description: {config.rxer_description}")
    logger.info(f"Receiver antenna type: {config.rxer_ant_type}")
    logger.info(f"Receiver number of antennas: {config.rxer_ant_num}")

    logger.info(f"Transmitter description: {config.txer_description}")
    logger.info(f"Transmitter description: {config.txer_ant_type}")
    logger.info(f"Transmitter description: {config.txer_ant_num}")
    
    logger.info(f"DOA Method: {config.doa_method}")
    logger.info(f"DOA Ns: {config.Ns}\n")
    

    # Evaluate metric for the configuration
    true_doa = np.rad2deg(config.theta0)
    estimated_doa = np.rad2deg(config.thetas[np.argmax(config.power_spectrum)])
    angle_error_doa = estimated_doa - true_doa

    true_noiseless_power = np.linalg.norm(config.txer.tx_signal)
    true_noisy_power = np.linalg.norm(config.rxer.rx_signal)
    estimated_power = np.linalg.norm(config.power_spectrum)
    l2_normal_error_power = np.linalg.norm(estimated_power - true_noiseless_power)
    l2_percent_error_power = (l2_normal_error_power / true_noiseless_power) * 100

    logger.info(f"Configutation metrics: ")
    logger.info(f"DOA Truth = {true_doa} [Degrees]")
    logger.info(f"DOA Estimate = {estimated_doa} [Degrees]")
    logger.info(f"DOA Angle Error (Estimate - True) = {angle_error_doa} [Degrees]\n")
    
    logger.info(f"True Noiseless Signal Power= {true_noiseless_power}")
    logger.info(f"True Noisy Signal Power= {true_noisy_power}")
    logger.info(f"Estimated Signal Power = {estimated_power}")
    logger.info(f"Power l2 Error (Estimate - True) = {l2_normal_error_power}")
    logger.info(f"Power l2 Percent Error = {l2_percent_error_power}%\n")

    logger.info("Simulation run log finished.\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


# Simulation proper functions
def run_simulation_instance(config_parameters, suppress_plots, output_path, tree_code):
    # Create the config
    # simulation config objects automatically perform calculations
    config = simulation_config(config_parameters)

    # Logging
    log_file_starter = "simulation_log.txt"
    log_path = os.path.join(output_path, log_file_starter)
    log_config_simulation(config=config, log_path=log_path, tree_code=tree_code)

    # Plotting if desired
    if not(suppress_plots):
        fig_file_starter = "simulation_figures_"
        fig_path = os.path.join(output_path, fig_file_starter) + tree_code

        plt.figure()
        plt.plot(config.rxer.rx_signal[0,:].squeeze().real[0:200])
        plt.plot(config.rxer.rx_signal[1,:].squeeze().real[0:200])
        plt.plot(config.rxer.rx_signal[2,:].squeeze().real[0:200])
        format_fig_to_file("rxed_signals", fig_path)

        plt.figure()
        plt.plot(np.rad2deg(config.thetas),config.power_spectrum)
        format_fig_to_file("estimate_AOA", fig_path, config.doa_method)

def simulate_all_options(hide_plots, output_trail, instantiated_options, tree_code, possible_options):
    # Base case is that no args were passed in
    # In such case construct a simulation config and save the results of it
    if len(possible_options) == 0:
       given_config = run_simulation_instance( tuple(instantiated_options), hide_plots, output_trail, tree_code)
    
    # Recursive case is that args are still left
    # In such case iterate over all options for the 0th arg and for each iteration instantiate the option and call fn1 again witht he new 'concrete' options and no 1st arg
    else:
        option0 = possible_options[0]
        
        i1 = 0
        for iter in option0:
            i1 += 1
            
            tree_string = tree_code + str(i1) # add some sort of different naming scheme
            concrete_options = instantiated_options + [iter]
            simulate_all_options(hide_plots, output_trail, concrete_options, tree_string, possible_options[1:])


# Main function finds (and if unfound creates) a stack of simulations to perform
# It then pops individual simulations off the stack, runs them, and saves their output
# Can optionally suppress plot output if undesired
def main(hide_plots = False):
    # Add the paths to the directories involved in simulation output invariant to user's OS
    cwd = os.getcwd()

    output_folder = "output_files"
    output_path = os.path.join(cwd, output_folder)
    
    ## Set up the simulation options and loop through all possible configurations
    # The lists at the top are the only values that need modified, all else is fixed and computed from numpy arrays
    options_f0 = [0.5e9]
    options_Fs_f0_mult = [1, 1.5, 2, 10, 40] # from hopefully severe aliasing to no aliasing
    options_Nsamples = [1e5]

    options_thetas = [np.linspace(-np.pi/2,np.pi/2,501)]
    options_phis = [np.zeros(1)]
    
    options_rxer_description = [""]
    options_rxer_ant_type = ['ula']
    options_rxer_ant_num  = [7]

    # signal arrival
    options_theta0 = [np.deg2rad(15)]
    options_r0 = [20] # Units?
    
    # transmitter
    options_txer_description = [""]
    options_txer_ant_type = ['iso']
    options_txer_ant_num  = [-1] # antenna type is iso and doesnt use this currently
    
    options_doa_method = ["MVDR", "MUSIC", "DBT"]
    options_doa_extra = [1]

    options_list = [options_f0, options_Fs_f0_mult, options_Nsamples, 
                    options_thetas, options_phis, options_rxer_description, 
                    options_rxer_ant_type, options_rxer_ant_num, 
                    options_theta0, options_r0, options_txer_description, options_txer_ant_type, 
                    options_txer_ant_num, options_doa_method, options_doa_extra]

    simulate_all_options(hide_plots=suppress_plots, output_trail=output_path, 
        instantiated_options=[], tree_code="", possible_options=options_list)


# Runs main with adjustable settings when this file is run
if __name__ == '__main__':
    suppress_plots = False
    
    main(suppress_plots)