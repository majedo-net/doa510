import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from itertools import product


logger = logging.getLogger(__name__)
#plt.style.use(['science', 'ieee'])

class Angle():
    """
    Implements a class that automatically formats angles in degrees and radians which can be edited as needed
    """
    def __init__(self, inp_radians = "NAN", inp_degrees = "NAN"):
        if inp_radians == "NAN":
            if inp_degrees == "NAN":
                raise ValueError("Angle object received no argument for radians or degrees.")
            
            else:
                self.set_degrees(inp_degrees)

        else:
            if inp_degrees == "NAN":
                self.set_radians(inp_radians) 
                
            else:
                raise ValueError("Angle object received both an argument for radians and an argument for degrees.")      

    def set_radians(self, inp_radians):
        self.radians = inp_radians
        self.degrees = self.radians * (180 / np.pi)          

    def set_degrees(self, inp_degrees):
        self.degrees = inp_degrees
        self.radians = self.degrees * (np.pi / 180)        

    def get_radians(self):
        return self.radians

    def get_degrees(self):
        return self.degrees



def calculate_metrics_and_log(true_doa, estimated_doa, method_name, log_file=None, execution_time=None, convergence_epochs=None):
    """
    Calculate metrics for a DOA estimation method and optionally log to a file.

    Args:
        true_doa (list or np.array): True angles of arrival (in degrees).
        estimated_doa (list or np.array): Estimated angles of arrival (in degrees).
        method_name (str): Name of the DOA estimation method.
        log_file (file object): File object to write metrics to (optional).
        execution_time (float): Time taken to execute the method (in seconds).
        convergence_epochs (int): Number of epochs required to converge (DBT only).

    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """
    # Ensure inputs are numpy arrays for consistency
    true_doa = np.array(true_doa)
    estimated_doa = np.array(estimated_doa)
    
    # Calculate metrics
    mae = np.mean(np.abs(true_doa - estimated_doa))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((true_doa - estimated_doa) ** 2))  # Root Mean Squared Error
    bias = np.mean(estimated_doa - true_doa)  # Bias

    # Compile metrics into a dictionary
    metrics = {
        "Method": method_name,
        "MAE": mae,  # No unit attached here
        "RMSE": rmse,  # No unit attached here
        "Bias": bias,  # No unit attached here
        "Execution Time": execution_time if execution_time is not None else "N/A",
    }

    # Add convergence epochs only for DBT
    if method_name == "DBT" and convergence_epochs is not None:
        metrics["Convergence Epochs"] = convergence_epochs

    return metrics



def generate_plots(plot_data, output_folder, doa_method, simulation_index):
    """
    Generate and save plots for the specified plot types in the options.
    Args:
        plot_data: Dictionary containing additional data for plotting.
        output_folder: Directory to save the plots.
        doa_method: String representing the DOA method used.
        simulation_index: Index of the current simulation (1-based).
    """
    if doa_method == "DBT":
        # Plot transmitter trajectory
        file_suffix = f"{doa_method}_trajectory_{simulation_index}.png"
        filepath = os.path.join(output_folder, file_suffix)
        plt.scatter(plot_data["xs"], plot_data["ys"], label="Transmitter Trajectory")
        plt.scatter(plot_data["antennaPositions"][1, :], plot_data["antennaPositions"][0, :], label="Receiver Antenna Array")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Transmitter Trajectory")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

        # Plot DOA estimation accuracy
        file_suffix = f"{doa_method}_accuracy_{simulation_index}.png"
        filepath = os.path.join(output_folder, file_suffix)
        plt.plot(np.arange(len(plot_data["est_doa"])), plot_data["est_doa"], label="Estimated DOA")
        plt.plot(np.arange(len(plot_data["true_doa"])), plot_data["true_doa"], label="True DOA")
        plt.xlabel("Epoch")
        plt.ylabel("DOA (Degrees)")
        plt.title("DOA Estimation Accuracy and Convergence")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    elif doa_method in ["MUSIC", "MVDR"]:
        # Plot spectrum
        file_suffix = f"{doa_method}_spectrum_{simulation_index}.png"
        filepath = os.path.join(output_folder, file_suffix)
        plt.plot(np.rad2deg(plot_data["thetas"]), plot_data["Spectrum"], label="Spatial Spectrum")
        plt.axvline(x=(plot_data["true_doa"]), color="red", linestyle="--", label="True DOA")
        plt.xlabel("Angle (Degrees)")
        plt.ylabel("Power Spectrum (dB)")
        plt.title(f"{doa_method} Spatial Spectrum")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

        # Plot DOA accuracy comparison
        file_suffix = f"{doa_method}_accuracy_{simulation_index}.png"
        filepath = os.path.join(output_folder, file_suffix)
        plt.scatter([0], [(plot_data["estimated_doa"])], label="Estimated DOA", color="blue")
        plt.axhline(y=(plot_data["true_doa"]), color="red", linestyle="--", label="True DOA")
        plt.xlabel("Simulation Configuration")
        plt.ylabel("DOA (Degrees)")
        plt.title(f"{doa_method} DOA Accuracy Comparison")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

        # Plot noise robustness if multiple noise powers are defined
        if len(plot_data.get("noise_levels", [])) > 1:
            file_suffix = f"{doa_method}_noise_{simulation_index}.png"
            filepath = os.path.join(output_folder, file_suffix)
            plt.plot(plot_data["noise_levels"], plot_data["mae_values"], label="MAE vs Noise Power")
            plt.xlabel("Noise Power (dB)")
            plt.ylabel("Mean Absolute Error (Degrees)")
            plt.title(f"{doa_method} Noise Robustness")
            plt.legend()
            plt.savefig(filepath)
            plt.close()
            
        STOP

def log_simulation_details(config, log_path, results=None, sim_index=None, total_sims=None):
    """
    Logs the simulation configuration and results to a file.
    Args:
        config: SimulationConfig object with simulation settings.
        log_path: Filepath to save the log.
        results: Dictionary or details about the results.
        sim_index: Current simulation index (1-based).
        total_sims: Total number of simulations.
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_path, 'a') as log_file:
        log_file.write(f"Simulation {sim_index}/{total_sims}\n")
        log_file.write("=" * 40 + "\n")


        log_file.write(f"Simulation run started logging at: {datetime.now()}\n")

               # Log configuration details
        log_file.write("Configuration:\n")
        log_file.write(f"  Carrier Frequency (f0): {config.f0} Hz\n")
        log_file.write(f"  Sampling Frequency (Fs): {config.Fs} Hz\n")
        log_file.write(f"  Signal Type: {config.tx_signal_type}\n")
        if config.tx_signal_type == "bpsk":
            log_file.write(f"  Symbol Rate: {config.options.get('symbol_rate', 'N/A')} Hz\n")
        log_file.write(f"  Number of Samples: {config.Nsamples}\n")
        log_file.write(f"  Receiver Antenna Elements: {config.rxer_ant_num}\n")
        log_file.write(f"  Array Spacing: {config.array_spacing} wavelengths\n")
        log_file.write(f"  True DOA (theta0): {np.rad2deg(config.theta0):.2f} degrees\n")
        log_file.write(f"  Transmitter Distance (r0): {config.r0} meters\n")
        log_file.write(f"  Transmitter Antenna Elements: {config.txer_ant_num}\n")
        log_file.write(f"  DOA Method: {config.doa_method}\n")
        log_file.write(f"  Noise Power: {config.pnoise} dB\n")
        
        # Log DBT-specific parameters if applicable
        if config.doa_method == "DBT":
            log_file.write(f"  Number of Subarrays (Nsubarrays): {config.Nsubarrays}\n")
            log_file.write(f"  Number of Epochs (Nepochs): {config.Nepochs}\n")
        
        log_file.write("\nResults:\n")
        
        # Log performance metrics
        metrics = calculate_metrics_and_log(
            true_doa=[np.rad2deg(config.theta0)],  # true DOA
            estimated_doa=[results.get("estimated_doa", 0)],  # estimated DOA
            method_name=config.doa_method
        )

        log_file.write(f"  Execution Time: {results.get('Execution Time', 'N/A'):.2f} seconds\n")
        log_file.write(f"  Mean Absolute Error (MAE): {metrics['MAE']:.2f}\n")
        log_file.write(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f} \n")
        log_file.write(f"  Bias: {metrics['Bias']:.2f} \n")
        log_file.write(f"  True DOA: {np.rad2deg(config.theta0):.2f}\n")
        estimated_doa = results.get('estimated_doa', 'N/A')
        log_file.write(f"  Estimated DOA: {estimated_doa:.2f}\n" if isinstance(estimated_doa, (int, float)) else f"  Estimated DOA: {estimated_doa}\n")

        
        # Log DBT-specific performance metrics
        if config.doa_method == "DBT" and "Convergence Epochs" in metrics:
            log_file.write(f"  Convergence Epochs: {metrics['Convergence Epochs']}\n")
        

        log_file.write("\n")
        log_file.write("-" * 40 + "\n")


class SimulationConfig:
    """
    Configuration file for a simulation.
    Includes dynamic handling for DBT and static configurations for other methods.
    """
    def __init__(self, options): 

        # Store all options
        self.options = options  

        # DOA method 
        self.doa_method = options["doa_method"]

        # Store receiver antennae type and amount 
        self.rxer_ant_type = options["rxer_ant_type"]  
        self.rxer_ant_num = options["rxer_ant_num"]  


        # Store transmitter antennae type and amount 
        self.txer_ant_type = options["txer_ant_type"] 
        self.txer_ant_num = options["txer_ant_num"]  


        # Carrier frequency and sampling frequency
        self.f0 = options["f0"]
        self.lambda0 = 3e8 / self.f0
        self.Fs = options["Fs"]
        self.Nsamples = options["Nsamples"]

        # Noise 
        self.noise = options["noise_power"]
        
        # Array spacing
        self.array_spacing = options.get("array_spacing", self.lambda0 / 2)

        # Vector of scan angles
        self.thetas = options["thetas"]
        self.phis = options["phis"]

        # Receiver configuration
        self.rxer = receiver.Receiver()
        self.rxer_ant_type = options["rxer_ant_type"]
        self.rxer_ant_num = options["rxer_ant_num"]
        self.rxer.ant = antenna.Antenna(self.rxer_ant_type, d=self.array_spacing, N=self.rxer_ant_num)
        self.rxer.ant.computeVman(self.f0, self.thetas, self.phis)

        # Signal arrival configuration
        self.theta0 = options["theta0"]
        self.r0 = options["r0"]
        
        # Signal type selection
        self.tx_signal_type = options.get("tx_signal_type", "tone")  # default single-tone 
    
        # Transmitter configuration
        self.pos0 = self.r0 * np.asarray([np.cos(self.theta0), np.sin(self.theta0), 0])

        if self.doa_method == "DBT":
            # For DBT, include velocity (vel0) in the transmitter
            self.vel0 = options.get("vel0", np.asarray([0.0, 0.0, 0.0]))  # Default to zero velocity
            self.txer = transmitter.Transmitter(self.pos0, _v=self.vel0)
        else:
            # For other methods, initialize without velocity
            self.txer = transmitter.Transmitter(self.pos0)

        self.txer.ant = antenna.Antenna(options["txer_ant_type"])

        # Generate the transmit signal
        if self.tx_signal_type == "tone":
            self.txer.generateTone(self.f0, self.Fs, self.Nsamples)
        elif self.tx_signal_type == "bpsk":
            data = 2 * np.random.randint(0, 2, 32) - 1  # Generate 32 binary values
            symbol_rate = options.get("symbol_rate", 1e6)  
            self.txer.generateBPSK(self.f0, self.Fs, symbol_rate, data)
        else:
            raise ValueError(f"Unsupported signal type: {self.tx_signal_type}")

        # DOA estimation method
        self.Ns = options.get("doa_extra", -1) if self.doa_method == "MUSIC" else -1
        self.pnoise = options.get("noise_power", 0)

        # DBT-specific variables
        self.Nepochs = options.get("Nepochs", 100)
        self.dspace = self.array_spacing
        self.Nant = options["rxer_ant_num"]
        self.Nsubarrays = options.get("Nsubarrays", 2)
        self.Pmax = (self.Nant / self.Nsubarrays) ** 2 * self.Nsamples


    def run_dbt(self):
        """Run the DBT simulation with iterative updates."""
        th_guess = 0
        doa_error = 0
        est_doa = []
        true_doa = []
        xs = []
        ys = []

        for epoch in range(self.Nepochs):
            # Update transmitter position and receiver weights
            self.txer.timeStep(1)
            xs.append(self.txer.x[0])  
            ys.append(self.txer.x[1]) 

            # Iterate for multiple corrections within each epoch
            for _ in range(2):  

                # Update hybrid beamforming weights based on current estimate
                self.rxer.ant.hybridWeights(self.f0, th_guess, 0, Nsub=self.Nsubarrays, subtype='interleaved') 
            
                # Ensure rx_signal has the required shape for DBT
                if self.rxer.rx_signal is None or self.rxer.rx_signal.shape[0] < self.Nsubarrays:
                    raise ValueError("DBT requires at least two subarrays in rx_signal.")


                # Receive signal and compute DBT error
                self.rxer.receiveSignal(self.txer, self.thetas, self.phis, channel.awgn, noise_power=self.pnoise)
                doa_error = doa.DBT(self.rxer.rx_signal, self.Pmax, self.dspace / self.lambda0)
                
                # Update DOA estimate
                th_guess += doa_error
                if th_guess > np.pi:
                    th_guess = -np.pi
                elif th_guess < -np.pi:
                    th_guess = +np.pi
            
            # Log results for this epoch
            est_doa.append(float(np.rad2deg(th_guess)))
            true_doa.append(float(np.rad2deg(self.theta0)))

        # Return data needed for plotting
        plot_data = {
        "xs": xs,
        "ys": ys,
        "est_doa": est_doa,
        "true_doa": true_doa,
        "antennaPositions": self.rxer.ant.antennaPositions,
        }
        
        return plot_data

    def run(self):
        """Run the simulation for the selected DOA method."""
    
        import time

        # Ensure the receiver gets the signal
        self.rxer.receiveSignal(
            transmitter=self.txer,
            thetas=self.thetas,
            phis=self.phis,
            channel=channel.awgn,
            noise_power=self.pnoise
        )

         # Start timer for execution time metric 
        start_time = time.perf_counter()

        # Initialize dictionaries
        results = {}
        plot_data = {}  # store additional data needed for plots

            # Perform the DOA estimation
        if self.doa_method == "DBT":
            plot_data = self.run_dbt()  # Collect plot data from run_dbt
            execution_time = time.perf_counter() - start_time
            estimated_doa = plot_data["est_doa"][-1] if plot_data["est_doa"] else None

            # Save primary results
            results = {
                "Execution Time": execution_time,
                "estimated_doa": estimated_doa,
            }

        elif self.doa_method in ["MVDR", "MUSIC"]:
            if self.doa_method == "MVDR":
                spectrum = [
                    doa.power_mvdr(self.rxer.ant.vk[thi, :], self.rxer.rx_signal)
                    for thi in range(self.thetas.size)
                ]
            else:  # MUSIC
                spectrum = doa.MUSIC(self.rxer.ant.vk, self.rxer.rx_signal, Ns=self.Ns)

            execution_time = time.perf_counter() - start_time
            estimated_doa = np.rad2deg(self.thetas[np.argmax(spectrum)])
            true_doa = float(np.rad2deg(self.theta0))

            # Save primary results
            results = {
                "Spectrum": spectrum,
                "Execution Time": execution_time,
            }

            # Save additional data for plots
            plot_data = {
                "Spectrum": spectrum,
                "thetas": self.thetas,
                "true_doa": true_doa,
                "estimated_doa": estimated_doa,
                }
            
            # # Compute noise robustness if multiple noise levels exist
            # if self.testing == 'noise':
            #     print(f"self.noise: {self.noise}, type: {type(self.noise)}")
            #     noise_levels = noise_vals
            #     mae_values = []
            #     for noise in noise_levels:
            #         self.pnoise = noise
            #         self.rxer.receiveSignal(
            #             transmitter=self.txer,
            #             thetas=self.thetas,
            #             phis=self.phis,
            #             channel=channel.awgn,
            #             noise_power=self.pnoise
            #         )
            #         spectrum = [
            #             doa.power_mvdr(self.rxer.ant.vk[thi, :], self.rxer.rx_signal)
            #             for thi in range(self.thetas.size)
            #         ] if self.doa_method == "MVDR" else doa.MUSIC(self.rxer.ant.vk, self.rxer.rx_signal, Ns=self.Ns)
            #         estimated_doa = np.rad2deg(self.thetas[np.argmax(spectrum)])
            #         mae_values.append(np.abs(true_doa - estimated_doa))

            #     # Add robustness data
            #     plot_data["noise_levels"] = noise_levels
            #     plot_data["mae_values"] = mae_values
            # print(f"self.noise: {self.noise}, type: {type(self.noise)}")
        else:
            raise ValueError(f"Unknown DOA method: {self.doa_method}")

        return results, plot_data  # Return both results and plot data

def main():
    # Ensure output directory is created at startup 
    output_folder = "output_files"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define simulation options
    options = {
    ## Signal characteristics
    "f0": [0.5e9],  # Carrier frequencies
    "Fs": [10e9, 20e9],  # Sampling frequencies
    "Nsamples": [50000],  # Number of samples    
    "noise_power": [-10, 0],  # Noise power in dB
    "tx_signal_type": ["tone"],  # Signal types: 'tone' (single-tone), bpsk 
    "symbol_rate": [100e6],  # Symbol rate for BPSK signal generation
    
    ## Signal transmission
    # Transmitter antenna array
    "txer_ant_type": ["iso"],  # Transmitter antenna type
    "txer_ant_num": [1],  # Number of transmitters     

    # Transmitter positioning relative to receiver antenna
    "theta0": [np.deg2rad(45)],  # True AoA in radians
    "r0": [50],  # Transmitter distance from receiver

    ## Signal reception
    # Receiver antenna array
    "rxer_ant_type": ["ula"],  # Receiver antenna type
    "rxer_ant_num": [4],  # Number of antenna elements
    "array_spacing": [0.5],  # Array spacing in wavelength units    

    # Receiver scanning
    "thetas": np.linspace(-np.pi / 2, np.pi / 2, 501),  # Fixed range of AoAs
    "phis": np.zeros(1),  # Fixed azimuth angle - 1D

    ## DOA estimation
    "doa_method": ["MVDR"],  # DOA estimation methods: DBT, MUSIC, MVDR
    "doa_extra": [1, 2, 3],  # Valid values must be < rxer_ant_num

    "Nsubarrays": [2, 4],  # Number of subarrays for DBT
    "Nepochs": [100],  # Number of epochs for DBT 
    "vel0": [np.asarray([0, -0.2, 0])],  # initial velocity for DBT
    }

    for key, value in options.items():
        print(f"Parameter: {key}, Data Type: {type(value)}")


    # DOA method for dynamic file naming
    doa_method = options["doa_method"][0]  # Assuming a single value for simplicity
    log_path = os.path.join(output_folder, f"{doa_method}_simulations_log.txt")

    # Clear or initialize the log file
    with open(log_path, 'w') as log_file:
        log_file.write(f"DOA Method: {doa_method}\n")
        log_file.write(f"Simulation Log started at {datetime.now()}\n\n")

    # Declare parameters to be constant 
    fixed_params = {k: options[k] for k in ["thetas", "phis"]}

    # Declare parameters to vary 
    dynamic_options = {k: options[k] for k in options if k not in fixed_params}

    # Generate all combinations of dynamic options
    keys, values = zip(*dynamic_options.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Add fixed parameters to each configuration - UNCOMMENT AND REMOVE LINE 
    for config_dict in combinations:
        config_dict.update(fixed_params)

    # Initialize counters for each DOA method
    simulation_index_by_method = {method: 1 for method in options["doa_method"]}

    # Run simulations for all combinations
    for config_dict in combinations:
        config = SimulationConfig(config_dict)
        doa_method = config.doa_method

        print(f"Running {doa_method} simulation {simulation_index_by_method[doa_method]} of {len(combinations)}...")

        # Run the simulation and get results
        results, plot_data = config.run()

        # Generate and save plots
        generate_plots(
            plot_data=plot_data,
            output_folder=output_folder,
            doa_method=doa_method,
            simulation_index=simulation_index_by_method[doa_method],
        )

        # Save log and results
        log_simulation_details(
            config=config,
            log_path=log_path,
            results=results,
            sim_index=simulation_index_by_method[doa_method],
            total_sims=len(combinations)
        )

        # Increment the simulation index for this method
        simulation_index_by_method[doa_method] += 1

    print("All simulations completed.")

    STOP


if __name__ == '__main__':
    main()
