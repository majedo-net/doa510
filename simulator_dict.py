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
    def __init__(self, options):
        # Carrier frequency and sampling frequency
        self.f0 = options["f0"]
        self.lambda0 = 3e8 / self.f0
        self.Fs = options["Fs_f0_mult"]
        self.Nsamples = options["Nsamples"]

        # Vector of scan angles
        self.thetas = options["thetas"]
        self.phis = options["phis"]

        # Receiver configuration
        self.rxer_description = options["rxer_description"]
        match self.rxer_description:
            case "":
                self.rxer = receiver.Receiver()
            case _:
                raise Exception(f"Receiver type '{self.rxer_description}' not implemented.")

        self.rxer_ant_type = options["rxer_ant_type"]
        self.rxer_ant_num = options["rxer_ant_num"]
        if self.rxer_ant_type == 'iso':
            self.rxer.ant = antenna.Antenna('iso')
        else:
            self.rxer.ant = antenna.Antenna(self.rxer_ant_type, d=self.lambda0 / 2, N=self.rxer_ant_num)

        self.rxer.ant.computeVman(self.f0, self.thetas, self.phis)

        # Signal arrival configuration
        self.theta0 = options["theta0"]
        self.r0 = options["r0"]
        self.pos0 = self.r0 * np.asarray([np.cos(self.theta0), np.sin(self.theta0), 0])

        # Transmitter configuration
        self.txer_description = options["txer_description"]
        match self.txer_description:
            case "":
                self.txer = transmitter.Transmitter(self.pos0)
            case _:
                raise Exception(f"Transmitter type '{self.txer_description}' not implemented.")

        self.txer_ant_type = options["txer_ant_type"]
        self.txer_ant_num = options["txer_ant_num"]
        if self.txer_ant_type == 'iso':
            self.txer.ant = antenna.Antenna('iso')
        else:
            self.txer.ant = antenna.Antenna(self.txer_ant_type, d=self.lambda0 / 2, N=self.txer_ant_num)

        # DOA estimation method
        self.doa_method = options["doa_method"]
        self.Ns = options.get("doa_extra", -1) if self.doa_method == "MUSIC" else -1

        # Noise power (optional)
        self.pnoise = options.get("noise_power", 0)  # Default to 0 dB if not specified

        # Generate the transmit signal
        self.txer.generateTone(self.f0, self.Fs, Nsamples=self.Nsamples)

        # Receive the signal
        self.rxer.receiveSignal(self.txer, self.thetas, self.phis, channel.awgn, noise_power=self.pnoise)

        # Compute the DOA power spectrum
        match self.doa_method:
            case "MVDR":
                self.power_spectrum = [
                    doa.power_mvdr(self.rxer.ant.vk[thi, :], self.rxer.rx_signal)
                    for thi in range(self.thetas.size)
                ]
                self.power_spectrum = 10 * np.log10(np.real(self.power_spectrum))  # Convert to dB

            case "MUSIC":
                self.power_spectrum = doa.MUSIC(self.rxer.ant.vk, self.rxer.rx_signal, Ns=self.Ns)

            case _:
                raise Exception(f"Unknown DOA method '{self.doa_method}' specified.")


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

    #aliasing 
    logger.info(f"Sampling Frequency (Fs) = {config.Fs} Hz")
    logger.info(f"Aliased (Fs / f0 < 2): {'Yes' if config.Fs / config.f0 < 2 else 'No'}")

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
    logger.info(f"Noise Power = {config.pnoise} dB")


    logger.info("Simulation run log finished.\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


# Simulation proper functions
def run_simulation_instance(config_parameters, suppress_plots, output_path, tree_code):
    """
    Runs a single simulation instance using provided configuration parameters.

    Args:
        config_parameters (dict): Dictionary of simulation parameters.
        suppress_plots (bool): Whether to suppress plot generation.
        output_path (str): Directory to save logs and plots.
        tree_code (str): Unique identifier for this simulation instance.

    Returns:
        None
    """
    f0 = config_parameters["f0"]
    Fs_mult_list = config_parameters["Fs_f0_mult"]  # List of Fs multipliers

    # Ensure Fs_mult_list is a list
    if isinstance(Fs_mult_list, (int, float)):  # If it's a single value
        Fs_mult_list = [Fs_mult_list]

    Nsamples = config_parameters["Nsamples"]
    thetas = config_parameters["thetas"]
    phis = config_parameters["phis"]
    noise_power = config_parameters["noise_power"]
    rxer_ant_type = config_parameters["rxer_ant_type"]
    rxer_ant_num = config_parameters["rxer_ant_num"]
    txer_ant_type = config_parameters["txer_ant_type"]
    txer_ant_num = config_parameters["txer_ant_num"]
    doa_method = config_parameters["doa_method"]
    doa_extra = config_parameters["doa_extra"]

    # Create a single log file
    log_file_name = (
        f"simulation_log_f0-{f0:.1e}_rxer-{rxer_ant_type}_rxerNum-{rxer_ant_num}_"
        f"txer-{txer_ant_type}_txerNum-{txer_ant_num}_doa-{doa_method}_extra-{doa_extra}.txt"
    )
    log_path = os.path.join(output_path, log_file_name)

    with open(log_path, "w") as log_file:
        # Write header
        log_file.write("Simulation Log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Carrier Frequency (f0): {f0:.2e} Hz\n")
        log_file.write(f"Receiver Antenna: {config_parameters['rxer_ant_type']} (Num: {config_parameters['rxer_ant_num']})\n")
        log_file.write(f"Transmitter Antenna: {config_parameters['txer_ant_type']} (Num: {config_parameters['txer_ant_num']})\n")
        log_file.write(f"DOA Method: {config_parameters['doa_method']} | Extra: {config_parameters['doa_extra']}\n")
        log_file.write("=" * 80 + "\n\n")

        for idx, Fs_f0_mult in enumerate(Fs_mult_list):
            # Recalculate Fs for the current iteration
            Fs = Fs_f0_mult * f0

            # Create a configuration for this specific Fs
            config = simulation_config({
                "f0": f0,
                "Fs": Fs,
                "Fs_f0_mult": Fs_f0_mult,
                "Nsamples": Nsamples,
                "thetas": thetas,
                "phis": phis,
                "rxer_description": config_parameters["rxer_description"],
                "rxer_ant_type": config_parameters["rxer_ant_type"],
                "rxer_ant_num": config_parameters["rxer_ant_num"],
                "theta0": config_parameters["theta0"],
                "r0": config_parameters["r0"],
                "txer_description": config_parameters["txer_description"],
                "txer_ant_type": config_parameters["txer_ant_type"],
                "txer_ant_num": config_parameters["txer_ant_num"],
                "doa_method": config_parameters["doa_method"],
                "doa_extra": config_parameters["doa_extra"],
            })

            config.pnoise = noise_power
            aliasing_condition = Fs / f0 < 2
            alias_status = "Yes" if aliasing_condition else "No"

            # Log iteration details
            log_file.write(f"Iteration {idx + 1}: Fs Multiplier = {Fs_f0_mult}\n")
            log_file.write(f"  Sampling Frequency (Fs): {Fs:.2e} Hz\n")
            log_file.write(f"  Aliasing: {alias_status}\n")
            log_file.write(f"  Number of Samples: {Nsamples}\n")
            true_doa = np.rad2deg(config.theta0)
            estimated_doa = np.rad2deg(config.thetas[np.argmax(config.power_spectrum)])
            angle_error_doa = estimated_doa - true_doa
            log_file.write(f"  True DOA: {true_doa:.2f} degrees\n")
            log_file.write(f"  Estimated DOA: {estimated_doa:.2f} degrees\n")
            log_file.write(f"  DOA Error: {angle_error_doa:.2f} degrees\n")

            true_noiseless_power = np.linalg.norm(config.txer.tx_signal)
            true_noisy_power = np.linalg.norm(config.rxer.rx_signal)
            estimated_power = np.linalg.norm(config.power_spectrum)
            l2_normal_error_power = np.linalg.norm(estimated_power - true_noiseless_power)
            l2_percent_error_power = (l2_normal_error_power / true_noiseless_power) * 100
            log_file.write(f"  Signal Power (True): {true_noiseless_power:.2f}\n")
            log_file.write(f"  Signal Power (Estimated): {estimated_power:.2f}\n")
            log_file.write(f"  Power Error: {l2_normal_error_power:.2f} ({l2_percent_error_power:.2f}%)\n")
            log_file.write("-" * 80 + "\n")

            # Generate plots if needed
            if not suppress_plots:
                fig_file_prefix = f"fig_f0-{f0:.1e}_Fs-{Fs:.1e}_"
                fig_path = os.path.join(output_path, fig_file_prefix + tree_code)

                # Received Signals Plot
                plt.figure()
                for ant_idx in range(min(config.rxer.rx_signal.shape[0], 3)):
                    plt.plot(config.rxer.rx_signal[ant_idx, :].real[:200], label=f"Antenna {ant_idx + 1}")
                plt.title(f"Received Signals (Fs = {Fs:.1e} Hz)")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.savefig(f"{fig_path}_rxed_signals.png")
                plt.close()

                # AoA Estimation Plot
                plt.figure()
                plt.plot(np.rad2deg(config.thetas), config.power_spectrum)
                plt.title(f"AoA Estimation (Fs = {Fs:.1e} Hz, DOA: {doa_method})")
                plt.xlabel("Angle (Degrees)")
                plt.ylabel("Power Spectrum (dB)")
                plt.savefig(f"{fig_path}_aoa_estimation.png")
                plt.close()

        log_file.write("Simulation run completed.\n\n")


def simulate_all_options(hide_plots, output_trail, instantiated_options, tree_code, possible_options):
    if len(possible_options) == 0:
        required_keys = [
            "f0", "Fs_f0_mult", "Nsamples", "thetas", "phis", "rxer_description",
            "rxer_ant_type", "rxer_ant_num", "theta0", "r0", "txer_description",
            "txer_ant_type", "txer_ant_num", "doa_method", "doa_extra", "noise_power"
        ]
        missing_keys = [key for key in required_keys if key not in instantiated_options]
        if missing_keys:
            raise KeyError(f"Missing keys in simulation options: {missing_keys}")

        run_simulation_instance(instantiated_options, hide_plots, output_trail, tree_code)
        return

    key, values = list(possible_options.items())[0]
    remaining_options = dict(list(possible_options.items())[1:])

    for idx, value in enumerate(values):
        if key == "Fs_f0_mult" and not isinstance(value, list):
            value = [value]

        new_instantiated_options = {**instantiated_options, key: value}
        print(f"Current key: {key}, Value: {value}")
        print(f"Instantiated options: {new_instantiated_options}")
        new_tree_code = f"{tree_code}_{key[:3]}{idx}"
        simulate_all_options(hide_plots, output_trail, new_instantiated_options, new_tree_code, remaining_options)


# Main function finds (and if unfound creates) a stack of simulations to perform
# It then pops individual simulations off the stack, runs them, and saves their output
# Can optionally suppress plot output if undesired
def main(hide_plots = False):
    # Add the paths to the directories involved in simulation output invariant to user's OS

    cwd = os.getcwd()

    output_folder = "output_files"
    output_path = os.path.join(cwd, output_folder)

     # Ensure output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ## Set up the simulation options and loop through all possible configurations
    # The lists at the top are the only values that need modified, all else is fixed and computed from numpy arrays

    options_dict = {
        "f0": [0.5e9],
        "Fs_f0_mult": [1, 1.5, 2, 10, 40],  # severe aliasing to no aliasing
        "Nsamples": [1e5],
        "thetas": [np.linspace(-np.pi/2, np.pi/2, 501)],
        "phis": [np.zeros(1)],
        "rxer_description": [""],
        "rxer_ant_type": ['ula'],
        "rxer_ant_num": [7],
        # signal arrival 
        "theta0": [np.deg2rad(15)],
        "r0": [20],
        # transmitter 
        "txer_description": [""],
        "txer_ant_type": ['iso'],
        "txer_ant_num": [-1],  #  antenna type is iso and doesnt use this currently
        "doa_method": ["MVDR", "MUSIC", "DBT"],
        "doa_extra": [1],
        "noise_power": [-20, -10, 0, 10, 20],  # Noise levels in dB
    }

    # Debug: Print options dictionary
    print("Simulation options dictionary:")
    for key, values in options_dict.items():
        print(f"  {key}: {values}")

    # Run simulations for all configurations
    simulate_all_options(
        hide_plots=hide_plots,
        output_trail=output_path,
        instantiated_options={},  # Start with an empty dictionary
        tree_code="",  # Root of the configuration tree
        possible_options=options_dict  # Pass the options dictionary
    )

    print("All simulations completed. Results saved in:", output_path)

# Runs main with adjustable settings when this file is run
if __name__ == '__main__':
    suppress_plots = False
    
    main(suppress_plots)