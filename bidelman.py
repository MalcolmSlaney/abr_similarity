from absl import app
from absl import flags
import glob
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import os
from typing import Tuple


def read_bidelman_data(filename: str) -> np.ndarray:
    with open(filename) as fp:
      data = np.asarray(fp.read().split('\n'))
      data_list = []
      for line in data:
        data_list.append(np.asarray([float(f) for f in line.split(',')[:-1]]))
    return np.vstack(data_list[:-1])


def compute_spectrum_of_average(data: np.ndarray,
                     fs: float = 10000, # Sampling rate
                     plot: bool = False,
                     plot_title: str='Spectral Profile of Averaged Data') -> Tuple[np.ndarray, np.ndarray]:
  average_data = np.mean(data, axis=0)

  # Compute FFT
  n_samples = len(average_data)
  y_fft = np.fft.fft(average_data)

  # Compute magnitude spectrum in dB
  powerspectrum = np.abs(y_fft)
  dB_spectrum = 20 * np.log10(powerspectrum)

  # Create frequency array
  freq = np.fft.fftfreq(n_samples, d=1/fs)

  if plot:
    # Plot the spectral profile
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:n_samples//2], dB_spectrum[:n_samples//2]) # Plot only positive frequencies
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(plot_title)
    plt.grid(True)
    plt.show()

  return freq[:n_samples//2], dB_spectrum[:n_samples//2]


def compute_average_spectrum(data: np.ndarray,
                     fs: float = 10000, # Sampling rate
                     plot: bool = False,
                     plot_title: str='Average Spectral Profile Across All Trials') -> Tuple[np.ndarray, np.ndarray]:
  # Initialize a list to store spectral profiles for each row
  all_dB_spectra = []

  # Get the number of samples per row (trial)
  n_samples_per_row = data.shape[1]

  # Loop through each row of data
  for i in range(data.shape[0]):
      trial_data = data[i, :]

      # Compute FFT for the current trial
      y_fft_trial = np.fft.fft(trial_data)

      # Compute magnitude spectrum in dB
      powerspectrum_trial = np.abs(y_fft_trial)
      # Avoid log of zero, add a small epsilon if powerspectrum_trial contains zeros
      dB_spectrum_trial = 20 * np.log10(powerspectrum_trial + np.finfo(float).eps)

      all_dB_spectra.append(dB_spectrum_trial)

  # Convert list of spectra to a NumPy array
  all_dB_spectra_array = np.array(all_dB_spectra)

  # Average the spectral profiles across all trials
  average_dB_spectrum = np.mean(all_dB_spectra_array, axis=0)

  # Create frequency array (using n_samples_per_row for fftfreq)
  freq_per_row = np.fft.fftfreq(n_samples_per_row, d=1/fs)
  if plot:
    # Plot the averaged spectral profile
    plt.figure(figsize=(10, 6))
    plt.plot(freq_per_row[:n_samples_per_row//2], average_dB_spectrum[:n_samples_per_row//2]) # Plot only positive frequencies
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Average Magnitude (dB)')
    plt.title('Average Spectral Profile Across All Trials')
    plt.grid(True)
    plt.show()
  return freq_per_row[:n_samples_per_row//2], average_dB_spectrum[:n_samples_per_row//2]
  

def sum_power(freq_per_row, average_dB_spectrum, low_freq = 90, high_freq = 2000) -> float:

  # Ensure we only consider the positive frequency half of the spectrum
  positive_freq_indices = np.where(freq_per_row >= 0)[0]
  freq_positive = freq_per_row[positive_freq_indices]
  dB_spectrum_noise_positive = average_dB_spectrum[positive_freq_indices]

  # Find indices corresponding to the desired frequency band
  band_indices = np.where((freq_positive >= low_freq) & (freq_positive <= high_freq))

  # Extract the dB values within this band for the noise signal
  dB_values_in_band = dB_spectrum_noise_positive[band_indices]

  # Convert dB values to linear power (Power = 10^(dB/10))
  power_values_in_band = 10**(dB_values_in_band / 10)

  # Sum the linear power values to get the total power in the band
  total_noise_power_in_band = np.sum(power_values_in_band)

  return total_noise_power_in_band


def filter_noise(freq_per_row, signal_magnitude_spectrum, noise_magnitude_spectrum):  # 1. Create the filter by taking the absolute value of the signal spectrum and scaling it so the maximum is 1
  # powerspectrum_signal is already the linear magnitude spectrum (abs(FFT))
  # signal_magnitude_spectrum = powerspectrum_signal
  filter_scaling = signal_magnitude_spectrum / np.max(signal_magnitude_spectrum)

  # 2. Multiply the noise spectrum by this filter
  # average_dB_spectrum is in dB magnitude. Convert it to linear magnitude for multiplication.
  # noise_dB_spectrum = average_dB_spectrum
  # noise_magnitude_spectrum = 10**(noise_dB_spectrum / 20) # mag = 10^(dB_mag / 20)

  filtered_noise_magnitude_spectrum = noise_magnitude_spectrum * filter_scaling

  # 3. Calculate how much power is there in the filtered noise between 90Hz and 2000Hz
  low_freq = 90
  high_freq = 2000
  fs = 10000 # Sampling rate (already defined, but good to ensure context)

  # Ensure we only consider the positive frequency half of the spectrum for the filtered noise
  # freq_per_row corresponds to the frequencies for the noise spectrum, which was used for average_dB_spectrum
  positive_freq_indices_filtered = np.where(freq_per_row >= 0)[0]
  freq_positive_filtered = freq_per_row[positive_freq_indices_filtered]
  filtered_noise_magnitude_positive = filtered_noise_magnitude_spectrum[positive_freq_indices_filtered]

  # Find indices corresponding to the desired frequency band
  band_indices_filtered = np.where((freq_positive_filtered >= low_freq) & (freq_positive_filtered <= high_freq))

  # Extract the linear magnitude values within this band for the filtered noise
  magnitude_values_in_band_filtered = filtered_noise_magnitude_positive[band_indices_filtered]

  # Convert linear magnitude values to linear power (Power = magnitude^2) and sum them
  power_values_in_band_filtered = magnitude_values_in_band_filtered**2
  total_filtered_noise_power_in_band = np.sum(power_values_in_band_filtered)

  print(f"The total linear power in the filtered noise signal between {low_freq}Hz and {high_freq}Hz is: {total_filtered_noise_power_in_band:.4f}")

  return total_filtered_noise_power_in_band


def compute_advantage(curry_list) -> Tuple[NDArray, NDArray]:
  """Apply a bandpass filter to the noise spectrum and compute the power."""
  original_powers = []
  filtered_powers = []

  for file in curry_list:
    data = read_bidelman_data(file)
    freq, db_spectrum = compute_spectrum_of_average(data, plot=False)
    noise_freq, noise_db_spectrum = compute_average_spectrum(data, plot=False)

    original_noise_power = sum_power(noise_freq, noise_db_spectrum)
    new_noise_power = filter_noise(noise_freq, 10**(db_spectrum/20), 10**(noise_db_spectrum/20))

    original_powers.append(original_noise_power)
    filtered_powers.append(new_noise_power)
  return (np.array(original_powers), np.array(filtered_powers),
          freq, db_spectrum, noise_freq, noise_db_spectrum)


##################### Plotting Routine #####################

def plot_spectral_profiles(freqs, db_spectrum, noise_freqs, noise_db_spectrum, 
                           figsize=(6.4, 4.8),
                           plot_name: str = 'SpectralComparison.png',
                           plot_dir: str = '.'):
  plt.figure(figsize=figsize)
  plt.plot(freqs, db_spectrum, label='Signal')
  plt.plot(noise_freqs, noise_db_spectrum, label='Noise')

  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Spectral Profile: Signal vs. Noise')
  plt.legend()
  plt.grid(True)

  plt.axvline(90, ls='--')
  plt.axvline(2000, ls='--');
  plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)

def spectral_advantage_plot(freqs, original_powers, filtered_powers, 
                          db_spectrum, noise_freqs, noise_db_spectrum,
                          figsize=(6.4, 4.8), 
                          plot_file_name: str = 'SpectralAdvantageScatter.png',
                          plot_dir: str = '.')-> float:
  plt.figure(figsize=figsize)
  # plt.subplot(1, 2, 1)
  # plt.plot(freqs, db_spectrum, label='Signal')
  # plt.plot(noise_freqs, noise_db_spectrum, label='Noise')

  # plt.xlabel('Frequency (Hz)')
  # plt.ylabel('Magnitude (dB)')
  # plt.title('Spectral Profile: Signal vs. Noise')
  # plt.legend()
  # plt.grid(True)

  # plt.axvline(90, ls='--')
  # plt.axvline(2000, ls='--');

  # plt.subplot(1, 2, 2)
  plt.plot(10*np.log10(original_powers), 10*np.log10(filtered_powers), 'x')
  plt.xlabel('Original Power (dB)')
  plt.ylabel('Filtered Power (dB)')
  plt.title('Original vs. Filtered Power')
  plt.savefig(os.path.join(plot_dir, plot_file_name), dpi=300)
  
  plt.figure(figsize=figsize)
  power_reduction = 10*np.log10(original_powers) - 10*np.log10(filtered_powers)
  plt.plot(10*np.log10(original_powers), power_reduction, 'x')
  plt.xlabel('Original Noise Power (dB) for S1 Preparations')
  plt.ylabel('Noise Reduction (dB)')
  plt.title('Noise Reduction Due to Matched Filtering')
  plt.ylim(0, 5.5)
  plt.axhline(np.mean(power_reduction), ls='--', color='blue')
  plot_file_name = plot_file_name.replace('.png', '2.png')
  plt.savefig(os.path.join(plot_dir, plot_file_name), dpi=300)

  return np.mean(10*np.log10(original_powers) - 10*np.log10(filtered_powers))



flags.DEFINE_string('data_dir', 'Data/Bidelman', 'Directory containing Bidelman ABR data files.')
flags.DEFINE_string('plot_dir', 'Plots', 'Directory to save plots.')
FLAGS = flags.FLAGS

def main(_):
  file_pattern = os.path.join(FLAGS.data_dir, '*/*.txt')
  curry_list = glob.glob(file_pattern)

  (original_powers, filtered_powers, freqs, db_spectrum, 
   noise_freqs, noise_db_spectrum) = compute_advantage(curry_list)

  plot_spectral_profiles(freqs, db_spectrum, noise_freqs, noise_db_spectrum, 
                         plot_dir=FLAGS.plot_dir)

  spectral_advantage_plot(freqs, original_powers, filtered_powers, 
                          db_spectrum, noise_freqs, noise_db_spectrum,
                          plot_dir=FLAGS.plot_dir)


if __name__ == '__main__':
   app.run(main)