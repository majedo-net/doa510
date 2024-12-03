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
    Fs = 10e9  # oversample by a lot
    Nsamples = 10000

    # Vector of scan angles
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 501)
    phis = np.zeros(1)

    # Transmitter setup
    theta0 = np.deg2rad(60)  # True AoA
    r0 = 20
    pos0 = r0 * np.array([np.cos(theta0), np.sin(theta0), 0])
    txer = transmitter.Transmitter(pos0)
    txer.ant = antenna.Antenna('iso')
    data = 2 * np.random.randint(0, 2, 32) - 1
    txer.generateBPSK(f0, Fs, 100e6, data)

    # Configurations for varying number of array elements
    array_sizes = [4, 8, 16, 32]
    spectral_results = []
    subspace_results = []

    for Nant in array_sizes:
        rxer = receiver.Receiver()
        rxer.ant = antenna.Antenna('ula', d=base_dspace, N=Nant)
        rxer.ant.computeVman(f0, thetas, phis)

        # Receive signal
        rxer.rx_signal = None  # Reset received signal
        rxer.receiveSignal(txer, thetas, phis, channel.awgn, noise_power=0)

        # Spectral Estimation (MVDR)
        mvdr_spectrum = []
        for thi in range(thetas.size):
            mvdr_spectrum.append(doa.power_mvdr(rxer.ant.vk[thi, :], rxer.rx_signal))
        mvdr_spectrum = 10 * np.log10(np.real(mvdr_spectrum))
        spectral_results.append((Nant, mvdr_spectrum))

        # Subspace Method (MUSIC)
        music_spectrum = doa.MUSIC(rxer.ant.vk, rxer.rx_signal, Ns=1)
        subspace_results.append((Nant, music_spectrum))

    # Visualization and Saving
    for i, (Nant, mvdr_spectrum) in enumerate(spectral_results):
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(thetas), mvdr_spectrum, label=f"MVDR - {Nant} Antennas")
        ax.set_title(f"Spectral Method Sensitivity (MVDR) - {Nant} Elements")
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("MVDR Spectrum (dB)")
        ax.legend()
        save_plot(fig, f"sensitivity_mvdr_{Nant}_antennas.png")

    for i, (Nant, music_spectrum) in enumerate(subspace_results):
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(thetas), music_spectrum, label=f"MUSIC - {Nant} Antennas")
        ax.set_title(f"Subspace Method Sensitivity (MUSIC) - {Nant} Elements")
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("MUSIC Spectrum (dB)")
        ax.legend()
        save_plot(fig, f"sensitivity_music_{Nant}_antennas.png")

    # Combined Summary
    print("Summary of AoA Estimation Peaks:")
    for i, ((Nant, mvdr_spectrum), (_, music_spectrum)) in enumerate(zip(spectral_results, subspace_results)):
        mvdr_peak = np.rad2deg(thetas[np.argmax(mvdr_spectrum)])
        music_peak = np.rad2deg(thetas[np.argmax(music_spectrum)])
        print(f"{Nant} Antennas - MVDR Peak: {mvdr_peak:.2f}°, MUSIC Peak: {music_peak:.2f}°")
