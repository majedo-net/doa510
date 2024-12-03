import os
import numpy as np
import receiver, transmitter, antenna, doa, channel
import matplotlib.pyplot as plt
import scienceplots

# Use IEEE plot styles for high-quality figures
# Comment this line out if you need faster loading during development
# plt.style.use(['science', 'ieee'])

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

        # Receiver with uniform linear array antenna
        rxer = receiver.Receiver()
        rxer.ant = antenna.Antenna('ula', d=dspace, N=7)
        rxer.ant.computeVman(f0, thetas, phis)

        # Two transmitters at different AoA
        theta0 = np.deg2rad(45)
        theta1 = np.deg2rad(-10)
        r0 = 20
        pos0 = r0 * np.array([np.cos(theta0), np.sin(theta0), 0])
        pos1 = r0 * np.array([np.cos(theta1), np.sin(theta1), 0])

        txer0 = transmitter.Transmitter(pos0)
        txer0.ant = antenna.Antenna('iso')
        txer1 = transmitter.Transmitter(pos1)
        txer1.ant = antenna.Antenna('iso')

        # Generate the transmit signals
        txer0.generateTone(f0, Fs, Nsamples=Nsamples)
        txer1.generateTone(f0, Fs, Nsamples=Nsamples)

        # Receive the signals
        rxer.rx_signal = None  # Reset for clean signal addition
        rxer.receiveSignal(txer0, thetas, phis, channel.awgn, noise_power=-15)
        rxer.receiveSignal(txer1, thetas, phis, channel.awgn, noise_power=-15)

        # Plot received signals for visual inspection
        fig, ax = plt.subplots()
        ax.plot(rxer.rx_signal[0, :].squeeze().real[0:200], label="Antenna 1")
        ax.plot(rxer.rx_signal[1, :].squeeze().real[0:200], label="Antenna 2")
        ax.plot(rxer.rx_signal[2, :].squeeze().real[0:200], label="Antenna 3")
        ax.set_title(f"Ntx_ULA_MUSIC_aliasing - Received Signals - {config_name}")
        ax.legend()
        save_plot(fig, f"Ntx_ULA_MUSIC_aliasing_received_signals_{config_name}.png")

        # AoA Estimation using MUSIC
        power_spectrum = doa.MUSIC(rxer.ant.vk, rxer.rx_signal, Ns=2)
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(thetas), power_spectrum)
        ax.set_title(f"Ntx_ULA_MUSIC_aliasing - MUSIC AoA Estimation - {config_name}")
        ax.set_xlabel('Angle (Degrees)')
        ax.set_ylabel('Power Spectrum (dB)')
        save_plot(fig, f"Ntx_ULA_MUSIC_aliasing_aoa_estimation_{config_name}.png")

        # Estimate AoAs
        estimated_thetas = [np.rad2deg(thetas[i]) for i in np.argsort(power_spectrum)[-2:]]
        results[config_name] = estimated_thetas
        print(f"{config_name}: Estimated AoAs = {estimated_thetas}")

    # Summarize results
    print("\nFinal Results:")
    for config, est_aoas in results.items():
        print(f"{config}: Estimated AoAs = {est_aoas}")
