import os
import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# Use IEEE plot styles for high-quality figures
plt.style.use(['science', 'ieee'])
# Ensure TeX rendering is disabled
if "text.usetex" in plt.rcParams:
    plt.rcParams['text.usetex'] = False

def save_plot(fig, filename, output_dir="plots"):
    """Saves the figure to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")

if __name__ == '__main__':
    f0 = 0.5e9  # carrier frequency
    lambda0 = 3e8 / f0
    base_dspace = lambda0 / 2
    base_Fs = 20e9  # base oversampling frequency
    Nsamples = 10000

    # Vector of scan angles
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 501)
    phis = np.zeros(1)

    # Aliased configurations for testing
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

        # Receiver setup with uniform linear array antenna
        rxer = receiver.Receiver()
        rxer.ant = antenna.Antenna('ula', d=dspace, N=7)
        rxer.ant.computeVman(f0, thetas, phis)

        # Transmitter setup
        theta0 = np.deg2rad(45)
        r0 = 20
        pos0 = r0 * np.array([np.cos(theta0), np.sin(theta0), 0])
        txer = transmitter.Transmitter(pos0)
        txer.ant = antenna.Antenna('iso')

        # Generate the transmit signal
        txer.generateTone(f0, Fs, Nsamples=Nsamples)

        # Receive the signal
        rxer.receiveSignal(txer, thetas, phis, channel.awgn, noise_power=0)

        # MUSIC Power Spectrum Calculation
        power_spectrum = doa.MUSIC(rxer.ant.vk, rxer.rx_signal, Ns=1)

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(thetas), power_spectrum)
        ax.set_title(f"1tx_ULA_MUSIC_aliasing - AoA Estimation - {config_name}")
        ax.set_xlabel("Estimated AoA (theta)")
        ax.set_ylabel("MUSIC Spectrum (dB)")
        save_plot(fig, f"1tx_ULA_MUSIC_aliasing_{config_name}.png")

        doa_estimate = np.rad2deg(thetas[np.argmax(power_spectrum)])
        true_doa = np.rad2deg(theta0)
        error = np.abs(doa_estimate - true_doa)

        results[config_name] = error
        print(f"{config_name}: DOA Estimate = {doa_estimate:.2f}, Error = {error:.2f} degrees")

    # Summarize Results
    print("\nObserved DOA Errors:")
    for config, error in results.items():
        print(f"{config}: {error:.2f} degrees")
