import os
import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# Use IEEE plot styles for high quality figures
# Comment this line out if needed for faster development
# plt.style.use(['science', 'ieee'])

def save_plot(fig, filename, output_dir="plots"):
    """Saves the figure to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")

if __name__ == '__main__':
    f0 = 500e6  # carrier frequency
    lambda0 = 3e8 / f0
    base_dspace = lambda0 / 2
    base_Fs = 10e9  # base oversampling frequency
    Nsamples = 10000

    Nant = 16  # total number of antennas across all subarrays
    Nsubarrays = 2  # analog subarrays
    deff = (Nant / Nsubarrays) * base_dspace  # effective spacing of subarrays

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 501)
    phis = np.zeros(1)

    # Receiver and transmitter initialization
    rxer = receiver.Receiver()
    rxer.ant = antenna.Antenna('ula', d=base_dspace, N=Nant)
    rxer.ant.computeVman(f0, thetas, phis)

    theta0 = np.deg2rad(60)
    r0 = 20
    pos0 = r0 * np.array([np.cos(theta0), np.sin(theta0), 0])
    txer = transmitter.Transmitter(pos0)
    txer.ant = antenna.Antenna('iso')

    data = 2 * np.random.randint(0, 2, 32) - 1
    txer.generateBPSK(f0, base_Fs, 100e6, data)

    aliased_configs = {
        "Low Fs": base_Fs / 10,
        "High Fs": base_Fs,
        "Small dspace": base_dspace / 2,
        "Large dspace": base_dspace * 2
    }

    results = {}

    for config_name, config_value in aliased_configs.items():
        if "Fs" in config_name:
            Fs = config_value
            dspace = base_dspace
        else:
            Fs = base_Fs
            dspace = config_value

        rxer.ant = antenna.Antenna('ula', d=dspace, N=Nant)
        rxer.ant.computeVman(f0, thetas, phis)
        rxer.ant.hybridWeights(f0, 0, 0, Nsub=Nsubarrays, subtype='interleaved')
        rxer.receiveSignal(txer, thetas, phis, channel.awgn, noise_power=10)

        doa_error = doa.DBT(rxer.rx_signal, (Nant / Nsubarrays) ** 2 * Nsamples, dspace / lambda0)
        est_doa = []
        true_doa = []

        for _ in range(200):  # epochs
            txer.timeStep(1)
            rxer.ant.hybridWeights(f0, doa_error, 0, Nsub=Nsubarrays, subtype='interleaved')
            rxer.receiveSignal(txer, thetas, phis, channel.awgn, noise_power=10)
            doa_error = doa.DBT(rxer.rx_signal, (Nant / Nsubarrays) ** 2 * Nsamples, dspace / lambda0)
            est_doa.append(np.rad2deg(doa_error))
            true_doa.append(np.rad2deg(rxer.trueAoA(txer)))

        error = np.mean(np.abs(np.array(est_doa) - np.array(true_doa)))
        results[config_name] = error

        # Save the AoA estimation plot
        fig, ax = plt.subplots()
        ax.plot(est_doa, label="Estimated AoA")
        ax.plot(true_doa, label="True AoA")
        ax.set_title(f"1tx_ULA_DBT_aliasing - AoA Estimation - {config_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Angle (Degrees)")
        ax.legend()
        save_plot(fig, f"1tx_ULA_DBT_aliasing_aoa_estimation_{config_name}.png")

    # Print observed errors for analysis
    print("Observed AoA Errors:")
    for config, error in results.items():
        print(f"{config}: {error:.2f} degrees")
