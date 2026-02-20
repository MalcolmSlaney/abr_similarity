import absl.app
import absl.flags
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string('input', None, 'Input file path.')


def randomize_phase(waveform: NDArray) -> NDArray:
  # Input waveform is expected to be 2D: (N, num_waveforms)
  N, num_waveforms = waveform.shape

  # Perform FFT on the waveform (along the first axis, which is N)
  fft_waveform = np.fft.fft(waveform, axis=0)

  # Separate magnitude and phase
  magnitude = np.abs(fft_waveform)
  phase = np.angle(fft_waveform)

  randomized_phase_symmetric = np.zeros_like(phase, dtype=float)

  # Keep DC phase unchanged for each waveform
  randomized_phase_symmetric[0, :] = phase[0, :]
  if N % 2 == 0: # Handle Nyquist frequency if N is even
      randomized_phase_symmetric[N//2, :] = phase[N//2, :]

  # Randomize positive frequency phases
  pos_freq_indices = np.arange(1, (N + 1) // 2)
  random_angles_pos = np.random.uniform(-np.pi, np.pi, (len(pos_freq_indices), num_waveforms))
  randomized_phase_symmetric[pos_freq_indices, :] = random_angles_pos

  # Set negative frequency phases (conjugate symmetry)
  neg_freq_indices = np.arange(N // 2 + 1, N)
  # The slice [::-1, :] reverses rows, maintaining columns.
  randomized_phase_symmetric[neg_freq_indices, :] = -random_angles_pos[::-1, :]

  # Reconstruct the FFT result with original magnitude and randomized phase
  randomized_fft_waveform = magnitude * np.exp(1j * randomized_phase_symmetric)

  # Perform Inverse FFT to get the phase-randomized waveform
  phase_randomized_waveform = np.fft.ifft(randomized_fft_waveform, axis=0).real
  return phase_randomized_waveform

import numpy as np
from numpy.typing import NDArray

def calculate_jackknife_covariance(waveforms: NDArray) -> NDArray:
  """
  Calculates the jackknife covariance for a 2D array of waveforms.
  For each waveform, it calculates the covariance
  between that waveform and the average of all other waveforms.

  Args:
    waveforms (NDArray): A 2D NumPy array where rows represent samples
                         and columns represent different waveforms.

  Returns:
    NDArray: A 1D NumPy array containing the jackknife covariance coefficients
             for each waveform.
  """
  num_samples, num_waveforms = waveforms.shape

  if num_waveforms < 2:
      raise ValueError("Input array must contain at least two waveforms (columns) for jackknife correlation.")

  jackknife_covariances = np.zeros(num_waveforms)

  # Calculate the sum of all waveforms once for efficiency
  sum_of_all_waveforms = np.sum(waveforms, axis=1)

  for i in range(num_waveforms):
    # Current waveform is the target waveform
    target_waveform = waveforms[:, i]

    # Calculate the sum of all other waveforms
    sum_of_other_waveforms = sum_of_all_waveforms - target_waveform

    # Calculate the average of all other waveforms
    # Ensure division by at least 1, handle cases where num_waveforms - 1 might 
    # be zero (though handled by num_waveforms < 2 check)
    average_of_other_waveforms = sum_of_other_waveforms / (num_waveforms - 1)

    # Calculate covariance, Sum over time samples
    covariance =   np.sum (target_waveform * average_of_other_waveforms) 

    jackknife_covariances[i] = covariance

  return jackknife_covariances


def calculate_similarity(waveforms: NDArray) -> Tuple[NDArray, NDArray]:
  s_covariances = calculate_jackknife_covariance(waveforms)
  n_covariances = calculate_jackknife_covariance(randomize_phase(waveforms))
  return s_covariances, n_covariances


def main(argv):
  del argv  # Unused
  
  if FLAGS.input is None:
    raise absl.flags.ValidationError('--input flag is required.')
  
  # Read in the CSV data using numpy's loadtxt function
  try:
    my_data = np.loadtxt(FLAGS.input, delimiter=',')
    print(my_data)
  except IOError as e:
    print(f"Error reading CSV file: {e}")
  
  print(f'Found data with {my_data.shape[0]} samples and {my_data.shape[1]} trials.')

  s_covariances, n_covariances = calculate_similarity(my_data)

  dprime = (np.mean(s_covariances) - np.mean(n_covariances)) / np.sqrt(0.5 * (np.var(s_covariances) + np.var(n_covariances)))

  print(f'd-prime: {dprime}')

if __name__ == '__main__':
  absl.app.run(main)