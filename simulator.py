import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots
import logging
import os
import functools
import copy
from datetime import datetime

logger = logging.getLogger(__name__)

# use IEEE plot styles for high quality figures
# comment this line out and plots will load a bit faster for development work
plt.style.use(['science','ieee'])

# Simulation config to bundle all the associated simulation data and computations together
class simulation_config:
    """
    Configuration file for a simulation. 
    Holds all relevant starting settings for a given simulation run.
    """
    def __init__(self, *args):
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
        self.theta0_arr = options[8]
        self.r0_arr = options[9]
        self.pos0_arr = self.r0_arr*(np.asarray([np.cos(self.theta0_arr),np.sin(self.theta0_arr),0]))
        
        # transmitter
        self.txer_description = options[10]
        match self.txer_description:
            case "":
                self.txer = transmitter.Transmitter(self.pos0_arr)
            case _:
                raise Exception("transmitter type not implemented")
        
        self.txer_ant_type = options[11]
        self.txer_ant_num = options[12]
        if self.txer_ant_type == 'iso':
            self.txer.ant = antenna.Antenna('iso')
        else:
            self.txer.ant = antenna.Antenna(self.txer_ant_type,d=self.lambda0/2,N=self.txer_ant_num)

        self.aoa_method = options[13]

        if self.aoa_method == "MUSIC":
            self.Ns = options[14]
        else:
                self.Ns = -1 # not using a method that requires this

        # Acquire remaining derivable info from the info already in the config
        # generate the transmit signal
        self.txer.generateTone(self.f0,self.Fs,Nsamples=self.Nsamples)

        # receive the signal
        self.rxer.receiveSignal(self.txer,self.thetas,self.phis,channel.awgn, noise_power=options[15])        

        match self.aoa_method:
            case "MVDR":
                self.power_spectrum = []
                for thi in range(self.thetas.size):
                    self.power_spectrum.append(doa.power_mvdr(self.rxer.ant.vk[thi,:],self.rxer.rx_signal))
                self.power_spectrum = 10*np.log10(np.real(self.power_spectrum)) # convert to dB
                
            case "MUSIC":
                self.power_spectrum = doa.MUSIC(self.rxer.ant.vk,self.rxer.rx_signal,Ns=1)

            case _:
                raise Exception("Unknown AOA method")

class simulation_config_v2:
    """
    Configuration file for a simulation. 
    Holds all relevant starting settings for a given simulation run.
    """
    def __init__(self, config_dict):
        self.f0             = config_dict["f0"]
        self.lambda0        = 3e8 / self.f0
        self.Fs             = config_dict["Fs_f0_mult"] * self.f0
        self.Nsamples       = config_dict["Nsamples"]

        
        self.thetas         = config_dict["thetas"]
        self.phis           = config_dict["phis"]

        # receiver
        self.rxer_description   = config_dict["rxer_description"]
        match self.rxer_description:
            case "":
                self.rxer = receiver.Receiver()
            case _:
                raise Exception("transmitter type not implemented")
        
        self.rxer_ant_type      = config_dict["rxer_ant_type"]
        self.rxer_ant_num       = config_dict["rxer_ant_num"]
        
        if self.rxer_ant_type == 'iso':
            self.rxer.ant = antenna.Antenna('iso')
        else:
            self.rxer.ant = antenna.Antenna(self.rxer_ant_type, d=self.lambda0/2, N=self.rxer_ant_num)
        
        self.rxer.ant.computeVman(self.f0,self.thetas,self.phis)

        # signal arrival
        self.theta0_arr = config_dict["theta_arr"]
        self.r0_arr     = config_dict["r_arr"]
        self.pos0_arr   = np.nan # TODO fix this by producting the thet0 and r0 arrs according to the formula
        self.vel0_arr   = config_dict["v_arr"]
        self.acc0_arr   = config_dict["a_arr"]

        # transmitter
        self.txer_description   = config_dict["rxer_description"]
        match self.txer_description:
            case "":
                self.txer = transmitter.Transmitter(self.pos0_arr)
            case _:
                raise Exception("transmitter type not implemented")
        
        self.txer_ant_type      = config_dict["txer_ant_type"]
        self.txer_ant_num       = config_dict["txer_ant_num"]
        if self.txer_ant_type == 'iso':
            self.txer.ant = antenna.Antenna('iso')
        else:
            self.txer.ant = antenna.Antenna(self.txer_ant_type,d=self.lambda0/2,N=self.txer_ant_num)

        self.aoa_method = config_dict["AoA_method"]

        if self.aoa_method == "MUSIC":
            self.Ns = config_dict["AoA_MUSIC_extra"]
        else:
            self.Ns = -1 # not using a method that requires this

        # Acquire remaining derivable info from the info already in the config
        # generate the transmit signal
        self.txer.generateTone(self.f0,self.Fs,Nsamples=self.Nsamples)

        # receive the signal
        self.rxer.receiveSignal(self.txer,self.thetas,self.phis,channel.awgn, noise_power=config_dict["noise_power"])        

        match self.aoa_method:
            case "MVDR":
                self.power_spectrum = []
                for thi in range(self.thetas.size):
                    self.power_spectrum.append(doa.power_mvdr(self.rxer.ant.vk[thi,:],self.rxer.rx_signal))
                self.power_spectrum = 10*np.log10(np.real(self.power_spectrum)) # convert to dB
                
            case "MUSIC":
                self.power_spectrum = doa.MUSIC(self.rxer.ant.vk,self.rxer.rx_signal,Ns=1)

            case _:
                raise Exception("Unknown AOA method")


# Class to hold data to compare across a given metric
class metric_ensemble:
    def __init__(self):
        self.MVDR = []
        self.MUSIC = []
        self.DBT = []

# Plotting utility functions
def get_rxed_signals_plotting(fig_path):
    title_str = "Received Signals Sampled"
    xlabel_str = "Sample Number"
    ylabel_str = "Magnitude"
    path_str = fig_path + "_rxed_signals.png"

    plot_info_tuple = (title_str, xlabel_str, ylabel_str, path_str)

    return plot_info_tuple

def get_estimate_AoA_plotting(fig_path, AoA_method):
    title_str = "Estimated Angle of Arrival Using " + AoA_method
    xlabel_str = "Angle (Degrees)"
    ylabel_str = "AoA Metric"
    path_str = fig_path + "_rxed_signals.png"

    plot_info_tuple = (title_str, xlabel_str, ylabel_str, path_str)

    return plot_info_tuple   

def format_fig_to_file(plot_type, fig_trail, AOA_estimator = "not_used", *args):
    match plot_type:
        case "rxed_signals":
            title_str, xlabel_str, ylabel_str, path_str = get_rxed_signals_plotting(fig_trail)
        
        case "estimate_AOA":
            title_str, xlabel_str, ylabel_str, path_str = get_estimate_AoA_plotting(fig_trail, AOA_estimator)
        
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
    plt.close()


# Logging functions
def log_config_simulation(config, log_path):
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
    
    logger.info(f"AoA Method: {config.aoa_method}")
    logger.info(f"AoA Ns: {config.Ns}\n")
    

    # Evaluate metric for the configuration
    true_aoa = np.rad2deg(config.theta0_arr)
    estimated_aoa = np.rad2deg(config.thetas[np.argmax(config.power_spectrum)])
    angle_error_aoa = estimated_aoa - true_aoa

    true_noiseless_power = np.linalg.norm(config.txer.tx_signal)
    true_noisy_power = np.linalg.norm(config.rxer.rx_signal)
    estimated_power = np.linalg.norm(config.power_spectrum)
    l2_normal_error_power = np.linalg.norm(estimated_power - true_noiseless_power)
    l2_percent_error_power = (l2_normal_error_power / true_noiseless_power) * 100

    logger.info(f"Configutation metrics: ")
    logger.info(f"AoA Truth = {true_aoa} [Degrees]")
    logger.info(f"AoA Estimate = {estimated_aoa} [Degrees]")
    logger.info(f"AoA Angle Error (Estimate - True) = {angle_error_aoa} [Degrees]\n")
    
    logger.info(f"True Noiseless Signal Power= {true_noiseless_power}")
    logger.info(f"True Noisy Signal Power= {true_noisy_power}")
    logger.info(f"Estimated Signal Power = {estimated_power}")
    logger.info(f"Power l2 Error (Estimate - True) = {l2_normal_error_power}")
    logger.info(f"Power l2 Percent Error = {l2_percent_error_power}%\n")

    logger.info("Simulation run log finished.\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


# Simulation proper functions
def run_simulation_instance(config_parameters, suppress_plots, output_path, tree_code, aoa_ensemble):
    # Create the config
    # simulation config objects automatically perform calculations
    config = simulation_config(config_parameters)

    # Metrics
    true_aoa = np.rad2deg(config.theta0_arr)
    estimated_aoa = np.rad2deg(config.thetas[np.argmax(config.power_spectrum)])
    angle_error_aoa = estimated_aoa - true_aoa
    
    match config.aoa_method:
        case "MVDR":
            aoa_ensemble.MVDR.append(angle_error_aoa)
        case "MUSIC":
            aoa_ensemble.MUSIC.append(angle_error_aoa)
        case "DBT":
            aoa_ensemble.DBT.append(angle_error_aoa)

    # Logging
    log_file_starter = "simulation_log.txt"
    log_path = os.path.join(output_path, log_file_starter)
    log_config_simulation(config=config, log_path=log_path)

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
        format_fig_to_file("estimate_AOA", fig_path, config.aoa_method)
    
    
def simulate_all_options(hide_plots, output_trail, instantiated_options, tree_code, possible_options, aoa_ensemble):
    # Base case is that no args were passed in
    # In such case construct a simulation config and save the results of it
    if len(possible_options) == 0:
        simulation_results = run_simulation_instance( tuple(instantiated_options), hide_plots, output_trail, tree_code, aoa_ensemble)


    # Recursive case is that args are still left
    # In such case iterate over all options for the 0th arg and for each iteration instantiate the option and call fn1 again witht he new 'concrete' options and no 1st arg
    else:
        option0 = possible_options[0]
        
        i1 = 0
        for iter in option0:
            i1 += 1
            
            tree_string = tree_code + str(i1) # add some sort of different naming scheme
            concrete_options = instantiated_options + [iter]
            simulate_all_options(hide_plots, output_trail, concrete_options, tree_string, possible_options[1:], aoa_ensemble)


    
def simulate_all_options_v2(config_lookup, instantiated_dict, hide_plots, output_trail, tree_code, aoa_ensemble):
    # Base case is that the config lookup is empty
    # In such case construct a simulation config and save the results of it
    if len(config_lookup.keys()) == 0:
        simulation_results = run_simulation_instance_v2( instantiated_dict, hide_plots, output_trail, tree_code, aoa_ensemble)

    # Recursive case is that args are still left
    # In such case iterate over all options for the 0th arg and for each iteration instantiate the option and call fn1 again witht he new 'concrete' options and no 1st arg
    else:
        config_dict = copy.deepcopy(config_lookup)
        key, value = config_dict.popitem()

        for iter in value:
            i1 += 1
            
            tree_string = tree_code + str(i1) # add some sort of different naming scheme
            
            concrete_dict = copy.deepcopy(instantiated_dict)
            concrete_dict[key] = iter
            
            simulate_all_options(hide_plots, output_trail, concrete_dict, tree_string, config_lookup, aoa_ensemble)


# Main function finds (and if unfound creates) a stack of simulations to perform
# It then pops individual simulations off the stack, runs them, and saves their output
# Can optionally suppress plot output if undesired
def main(hide_plots = False):

    # Add the paths to the directories involved in simulation output invariant to user's OS
    cwd = os.getcwd()

    output_folder = "output_files"
    output_path = os.path.join(cwd, output_folder)
    
    ## Set up the simulation options and loop through all possible configurations
    #TODO USE A DICTIONARY INSTEAD OF A LIST OF OPTIONS
    config_dict = {}
    
    # Signal charcteristics
    config_dict["f0"]               = [0.5e9]
    config_dict["Fs_f0_mult"]       = [1, 1.5, 2, 10, 40]
    config_dict["Nsamples"]         = [1e5]
    config_dict["noise_power"]      = [-10, -3, 0, 3, 10, 13, 20] # dB

    # Angles to scan
    config_dict["thetas"]           = [np.linspace(-np.pi/2,np.pi/2,501)]
    config_dict["phis"]             = [np.zeros(1)]
    
    # Receiver
    config_dict["rxer_description"] = [""]
    config_dict["rxer_ant_type"]    = ['ula']
    config_dict["rxer_ant_num"]     = [7]

    # Positioning
    config_dict["theta_arr"]        = [np.deg2rad(15)]
    config_dict["r_arr"]            = [20]
    config_dict["v_arr"]            = [0]
    config_dict["a_arr"]            = [0]

    # Transmitter
    config_dict["txer_description"] = [""]
    config_dict["txer_ant_type"]    = ['iso']
    config_dict["txer_ant_num"]     = [-1]

    # Angle of Arrival Estimator
    config_dict["AoA_method"]       = ["MVDR", "MUSIC"]
    config_dict["AoA_MUSIC_extra"]  = [1]
    #config_dict["AoA_DBT_extra"]    = 


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
    
    options_AoA_method = ["MVDR", "MUSIC"]
    options_AoA_extra = [1]

    options_noise_power = [-10, -3, 0, 3, 10, 13, 20] # dB

    options_list = [options_f0, options_Fs_f0_mult, options_Nsamples, 
                    options_thetas, options_phis, options_rxer_description, 
                    options_rxer_ant_type, options_rxer_ant_num, 
                    options_theta0, options_r0, options_txer_description, options_txer_ant_type, 
                    options_txer_ant_num, options_AoA_method, options_AoA_extra, options_noise_power]

    aoa_error_data = metric_ensemble()

    simulate_all_options(hide_plots=hide_plots, output_trail=output_path, 
        instantiated_options=[], tree_code="", possible_options=options_list, aoa_ensemble=aoa_error_data)

    aoa_l1_errors   = metric_ensemble()    
    aoa_l1_errors.MVDR  = np.linalg.norm(aoa_error_data.MVDR, 1)
    aoa_l1_errors.MUSIC = np.linalg.norm(aoa_error_data.MUSIC, 1)
    #aoa_l1_errors.DBT   = np.linalg.norm(aoa_error_data.DBT, 1)
    
    aoa_l2_errors   = metric_ensemble()
    aoa_l2_errors.MVDR  = np.linalg.norm(aoa_error_data.MVDR, 2)
    aoa_l2_errors.MUSIC = np.linalg.norm(aoa_error_data.MUSIC, 2)
    #aoa_l2_errors.DBT   = np.linalg.norm(aoa_error_data.DBT, 2)

    aoa_lINF_errors = metric_ensemble()
    aoa_lINF_errors.MVDR  = np.linalg.norm(aoa_error_data.MVDR, np.inf)
    aoa_lINF_errors.MUSIC = np.linalg.norm(aoa_error_data.MUSIC, np.inf)
    #aoa_lINF_errors.DBT   = np.linalg.norm(aoa_error_data.DBT, np.inf)

    fig_path = os.path.join(output_path, "AoA_error")

    plt.figure()
    plt.plot(aoa_error_data.MVDR, label="MVDR")
    plt.plot(aoa_error_data.MUSIC, label="MUSIC")
    #plt.plot(aoa_error_data.DBT, label="DBT")
    plt.title("AoA Error Across Metrics")
    plt.xlabel("Configuration Count")
    plt.ylabel("Error (Estimated - True) [Degrees]")
    plt.legend()
    plt.savefig(fig_path + "_metrics")
    plt.close()

    plt.figure()
    plt.hist(aoa_error_data.MVDR)
    plt.title("MVDR AOA Error")
    plt.xlabel("Error (Estimated - True) [Degrees]")
    plt.ylabel("Count")
    plt.savefig(fig_path + "_MVDR")
    plt.close()    

    plt.figure()
    plt.hist(aoa_error_data.MUSIC)
    plt.title("MUSIC AOA Error")
    plt.xlabel("Error (Estimated - True) [Degrees]")
    plt.ylabel("Count")
    plt.savefig(fig_path + "_MUSIC")
    plt.close()

# Runs main with adjustable settings when this file is run
if __name__ == '__main__':
    suppress_plots = False
    
    main(suppress_plots)