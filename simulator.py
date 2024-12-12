import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import logging
import os
import scipy
from datetime import datetime
from itertools import product


logger = logging.getLogger(__name__)
#plt.style.use(['science', 'ieee'])

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

def generate_individual_run_plots(plot_data, output_folder, doa_method, simulation_index):
    """
    Generate and save plots for the specified plot types in the options.
    Args:
        plot_data: Dictionary containing additional data for plotting.
        output_folder: Directory to save the plots.
        doa_method: String representing the DOA method used.
        simulation_index: Index of the current simulation (1-based).
    """
    # Plot transmitter trajectory
    file_suffix = f"{doa_method}_trajectory_{simulation_index}.png"
    filepath = os.path.join(output_folder, file_suffix)
    cs = np.arange(len(plot_data["xs"])) * 256 / np.max(plot_data["xs"])   
    plt.scatter(plot_data["xs"], plot_data["ys"], c=cs, label="Transmitter Trajectory (From Dark to Light)")
    plt.scatter(plot_data["antennaPositions"][1, :], plot_data["antennaPositions"][0, :], label="Receiver Antenna Array")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Transmitter Trajectory for {doa_method}")
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
    plt.title(f"DOA Estimation Accuracy and Convergence for {doa_method}")
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def generate_statistical_output(aggregated_metrics, output_folder, doa_method, simulation_index, config, log_path, total_sims, num_trials):
    """
    Generate and save plots and statistics of the given run
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_path, 'a') as log_file:
        log_file.write(f"Simulation {simulation_index}/{total_sims}\n")
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

        # Log MUSIC-specific parameters if applicable
        if config.doa_method == "MUSIC":
            log_file.write(f"  Number of Signals Expected (MUSIC): {config.MUSIC_Ns}\n")

        # Log DBT-specific parameters if applicable
        if config.doa_method == "DBT":
            log_file.write(f"  Number of Subarrays (Nsubarrays): {config.Nsubarrays}\n")
            log_file.write(f"  Number of Epochs (Nepochs): {config.Nepochs}\n")
            log_file.write(f"  Number of Iterations: {config.num_iter}\n")        

        log_file.write("\nResults:\n")
            
        for key in aggregated_metrics:
            metric_description = scipy.stats.describe(aggregated_metrics[key])
            
            # Log statistical performance metrics
            log_file.write("\n" + "." * 40 + "\n")   
            log_file.write(f"  Metric: {key}\n")
            log_file.write(f"  Number of Obervations: {metric_description.nobs}\n")
            log_file.write(f"  Minimum Value: {metric_description.minmax[0]}\n")
            log_file.write(f"  Maximum Value: {metric_description.minmax[1]}\n")
            log_file.write(f"  Mean: {metric_description.mean}\n")
            log_file.write(f"  Standard Deviation: {np.sqrt(metric_description.variance)}\n")
            log_file.write(f"  Skewness: {metric_description.skewness}\n")
            log_file.write(f"  Kurtosis: {metric_description.kurtosis}\n")   

        log_file.write("\n")
        log_file.write("-" * 40 + "\n")        

    for key in aggregated_metrics:
        file_suffix = f"{doa_method}_{simulation_index}_{key}_hist.png"
        filepath = os.path.join(output_folder, file_suffix)
        plt.hist(aggregated_metrics[key])   
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.title(f"{key} Histogram Across {num_trials} Trials for {doa_method}")
        plt.savefig(filepath)
        plt.close() 

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

        # Log MUSIC-specific parameters if applicable
        if config.doa_method == "MUSIC":
            log_file.write(f"  Number of Signals Expected (MUSIC): {config.MUSIC_Ns}\n")

        # Log DBT-specific parameters if applicable
        if config.doa_method == "DBT":
            log_file.write(f"  Number of Subarrays (Nsubarrays): {config.Nsubarrays}\n")
            log_file.write(f"  Number of Epochs (Nepochs): {config.Nepochs}\n")
            log_file.write(f"  Number of Iterations: {config.num_iter}\n")     
            
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
        self.txer_ant_type = options["txer_ant_type"] 
        self.txer_ant_num = options["txer_ant_num"]          

        self.pos0 = self.r0 * np.asarray([np.cos(self.theta0), np.sin(self.theta0), 0])
        self.vel0 = options.get("vel0", np.asarray([0.0, 0.0, 0.0]))  # Default to zero velocity
        self.acc0 = options.get("acc0", np.asarray([0.0, 0.0, 0.0]))  # Default to zero acceleration
        self.txer = transmitter.Transmitter(self.pos0, _v=self.vel0, _a=self.acc0)

        self.txer.ant = antenna.Antenna(self.txer_ant_type)

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
        self.MUSIC_Ns = options.get("MUSIC_Ns", -1) if self.doa_method == "MUSIC" else -1
        self.pnoise = options.get("noise_power", 0)

        # DBT-specific variables
        self.Nepochs = options.get("Nepochs", 100)
        self.dspace = self.array_spacing
        self.Nant = options["rxer_ant_num"]
        self.Nsubarrays = options.get("Nsubarrays", 2)
        self.Pmax = (self.Nant / self.Nsubarrays) ** 2 * self.Nsamples
        self.num_iter = options.get("num_iter", 1)

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

            if self.doa_method == "DBT":
                # Iterate for multiple corrections within each epoch <- Why? Is the DBT quick enough to do this?
                for _ in range(self.num_iter):  
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
                    
            elif self.doa_method == "MUSIC":
                self.rxer.receiveSignal(self.txer,self.thetas,self.phis,channel.awgn,noise_power=self.pnoise)
                spectrum = doa.MUSIC(self.rxer.ant.vk, self.rxer.rx_signal, Ns=self.MUSIC_Ns)
                th_guess = self.thetas[np.argmax(spectrum)]

            elif self.doa_method == "MVDR":
                self.rxer.receiveSignal(self.txer,self.thetas,self.phis,channel.awgn,noise_power=self.pnoise)
                spectrum = [
                    doa.power_mvdr(self.rxer.ant.vk[thi, :], self.rxer.rx_signal)
                    for thi in range(self.thetas.size)
                ]
                
                th_guess = self.thetas[np.argmax(spectrum)]

            else:
                raise ValueError(f"Invalid doa_method of {self.doa_method}")
            
            # Log results for this epoch
            est_doa.append(float(np.rad2deg(th_guess)))
            true_doa.append(np.rad2deg(self.rxer.trueAoA(self.txer)))

        # Return data needed for plotting
        plot_data = {
        "xs": xs,
        "ys": ys,
        "est_doa": est_doa,
        "true_doa": true_doa,
        "antennaPositions": self.rxer.ant.antennaPositions,
        }
        
        execution_time = time.perf_counter() - start_time
        estimated_doa = plot_data["est_doa"][-1] if plot_data["est_doa"] else None

        # Save primary results
        results = {
            "Execution Time": execution_time,
            "estimated_doa": estimated_doa,
        }

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
    "Fs": [20e9],  # Sampling frequencies
    "Nsamples": [50000],  # Number of samples    
    "noise_power": [-10],  # Noise power in dB
    "tx_signal_type": ["tone"],  # Signal types: 'tone' (single-tone), bpsk 
    "symbol_rate": [100e6],  # Symbol rate for BPSK signal generation
    
    ## Signal transmission
    # Transmitter antenna array
    "txer_ant_type": ["iso"],  # Transmitter antenna type
    "txer_ant_num": [1],  # Number of transmitters     

    # Transmitter positioning relative to receiver antenna
    "theta0": [np.deg2rad(45)],  # True AoA in radians
    "r0": [50],  # Transmitter distance from receiver
    "vel0": [np.asarray([0, -0.2, 0])],  # initial velocity for DBT
    "acc0": [np.asarray([0, 0, 0])],    

    ## Signal reception
    # Receiver antenna array
    "rxer_ant_type": ["ula"],  # Receiver antenna type
    "rxer_ant_num": [4],  # Number of antenna elements
    "array_spacing": [0.5],  # Array spacing in wavelength units    

    # Receiver scanning
    "thetas": np.linspace(-np.pi / 2, np.pi / 2, 501),  # Fixed range of AoAs
    "phis": np.zeros(1),  # Fixed azimuth angle - 1D

    ## DOA estimation
    "doa_method": ["DBT", "MUSIC", "MVDR"],  # DOA estimation methods: DBT, MUSIC, MVDR
    
    # MUSIC exclusive
    "MUSIC_Ns": [1],#[1, 2, 3],  # Valid values must be < rxer_ant_num, represents the # of signals to search for

    # DBT exclusive
    "Nsubarrays": [2],#[2, 4],  # Number of subarrays for DBT
    "Nepochs": [100],  # Number of epochs for DBT 
    "num_iter": [1, 10],

    # Special
    "num_trials": 30, # This is constant across trials and does not vary
    "output_stats": True,
    "output_log": True,
    "output_plots": True,
    }

    for key, value in options.items():
        print(f"Parameter: {key}, Data Type: {type(value)}")

    # Initialize counters for each DOA method
    simulation_index_by_method = {method: 1 for method in options["doa_method"]}

    # DOA method for dynamic file naming
    for doa_method in options["doa_method"]:
        if options["output_log"]:
            log_path = os.path.join(output_folder, f"{doa_method}_simulations_log.txt")

            # Clear or initialize the log file
            with open(log_path, 'w') as log_file:
                log_file.write(f"DOA Method: {doa_method}\n")
                log_file.write(f"Simulation Log started at {datetime.now()}\n\n")

        if options["output_stats"]:
            # Do the same for the statistics log
            stats_path = os.path.join(output_folder, f"{doa_method}_statistics_log.txt")

            # Clear or initialize the log file
            with open(stats_path, 'w') as stats_file:
                stats_file.write(f"DOA Method: {doa_method}\n")
                stats_file.write(f"Simulation Log started at {datetime.now()}\n\n")

        # Declare parameters to be constant 
        fixed_params = {k: options[k] for k in ["thetas", "phis", "num_trials", "output_stats", "output_log", "output_plots"]}

        # Declare parameters to vary 
        dynamic_options = {k: options[k] for k in options if k not in fixed_params}

        # Remove doa_method from varying
        del dynamic_options["doa_method"]

        # Generate all combinations of dynamic options
        keys, values = zip(*dynamic_options.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        # Add fixed parameters and doa_method to each configuration - UNCOMMENT AND REMOVE LINE 
        for config_dict in combinations:
            config_dict.update(fixed_params)
            config_dict.update({"doa_method": doa_method})

        # Run simulations for all combinations
        for config_dict in combinations:

            print(f"Running {doa_method} simulation {simulation_index_by_method[doa_method]} of {len(combinations)}...")

            collected_metrics = {
                "MAE": [],  # No unit attached here
                "RMSE": [],  # No unit attached here
                "Bias": [],  # No unit attached here
                "Execution Time": [],
            }

            for _ in range(options["num_trials"]):
                # Generate, run the simulation, and get results
                config = SimulationConfig(config_dict)
                results, plot_data = config.run()

                # Log performance metrics
                trial_metrics = calculate_metrics_and_log(
                    true_doa=[np.rad2deg(config.theta0)],  # true DOA
                    estimated_doa=[results.get("estimated_doa", 0)],  # estimated DOA
                    method_name=config.doa_method,
                    execution_time=results["Execution Time"],
                )
                
                for key in collected_metrics:
                    collected_metrics[key].append(trial_metrics[key])

            # Generate and save plots from the data of the last run
            if options["output_plots"]:
                generate_individual_run_plots(
                    plot_data=plot_data,
                    output_folder=output_folder,
                    doa_method=doa_method,
                    simulation_index=simulation_index_by_method[doa_method],
                )

            # Save log and results
            if options["output_log"]:
                log_simulation_details(
                    config=config,
                    log_path=log_path,
                    results=results,
                    sim_index=simulation_index_by_method[doa_method],
                    total_sims=len(combinations)
                )
            
            # Analyze statistically the output data
            if options["output_stats"]:
                generate_statistical_output(
                    aggregated_metrics=collected_metrics,
                    output_folder=output_folder,
                    doa_method=doa_method,
                    simulation_index=simulation_index_by_method[doa_method],
                    
                    config=config,
                    log_path=stats_path,
                    total_sims=len(combinations),
                    
                    num_trials=options["num_trials"],
                )

            # Increment the simulation index for this method
            simulation_index_by_method[doa_method] += 1

    print("All simulations completed.")

if __name__ == '__main__':
    main()
