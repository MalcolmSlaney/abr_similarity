import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from analyze import randomize_phase, calculate_jackknife_covariance, calculate_similarity

from absl import app
from absl import flags
from absl.testing import absltest



def colored_theory_mean(signal_spectrogram, noise_spectrogram):
  assert signal_spectrogram.shape == noise_spectrogram.shape
  num_samples = signal_spectrogram.shape[0]
  # Sum across frequency bins
  return np.sum(signal_spectrogram*np.conjugate(signal_spectrogram))/num_samples

def colored_theory_var(signal_spectrogram, noise_spectrogram):
  assert signal_spectrogram.shape == noise_spectrogram.shape
  num_samples = signal_spectrogram.shape[0]
  # Sum over frequency bins
  return np.sum(signal_spectrogram*np.conjugate(signal_spectrogram)*
                noise_spectrogram*np.conjugate(noise_spectrogram))/num_samples**2

def colored_theory_dprime(signal_spectrogram, noise_spectrogram):
  mean = colored_theory_mean(signal_spectrogram, noise_spectrogram) - colored_theory_mean(0*signal_spectrogram, noise_spectrogram)
  var = colored_theory_var(signal_spectrogram, noise_spectrogram) + colored_theory_var(0*signal_spectrogram, noise_spectrogram)
  return mean/np.sqrt(var)



def make_hermitian(s: NDArray) -> NDArray:
  """Flip the positive frequency part of the spectrum (first half) to get the
  negative frequencies (2nd half)
  """
  N = (len(s)-1)*2
  assert np.floor(np.log2(N)) == np.log2(N)
  r = np.zeros(N, dtype=complex)
  r[:len(s)] = s
  r[len(s):] = np.conjugate(np.flipud(s[1:-1]))
  return r

def make_noise(N: int, noise_level: float, flat=True) -> NDArray:
  """Create a noise spectrum and invert it to get a real noisy waveform.
  Strictly speaking this is a "random phase white noise process" since the 
  spectrum is a constant. (But it's still white noise because the phases are
  uncorrelated so by the law of large numbers the resulting waveform is 
  Gaussian.)
  """
  if flat:
    spectrum = N * noise_level * np.exp(1j * np.random.rand(N//2+1)*2*np.pi)
    spectrum[0] = 0*noise_level
    spectrum[N//2] = 0*noise_level
    spectrum = make_hermitian(spectrum)
    return spectrum, np.real(np.fft.ifft(spectrum))
  else:
    noise = np.random.randn(N) * noise_level
    return np.abs(np.fft.fft(noise)), noise

def make_signal(spectrum: NDArray) -> NDArray:
  """Given a Fourier spectrum, invert the FFT to get a real time-domain waveform."""
  # num_samples = len(spectrum)
  return np.real(np.fft.ifft(spectrum))

def create_sinusoid_abr(num_samples = 2048, num_waveforms = 1024,
                        signal_level = 1, noise_level = 10, flat=True):
  s = signal_level*np.sin(2*np.pi*4*np.arange(num_samples)/num_samples)

  # waveforms = np.expand_dims(s, 1) + np.random.randn(num_samples, num_waveforms) * noise_level
  waveforms = np.zeros((num_samples, num_waveforms))
  for i in range(num_waveforms):
    _, n = make_noise(num_samples, noise_level, flat=flat)
    waveforms[:, i] = s + n
  return waveforms

def make_experiment(signal_waveform: NDArray,
                    num_trials: int,
                    noise_level: float = 1,
                    flat=True) -> Tuple[NDArray, NDArray]:
  """Create a set of trials that make up a full experiment."""
  num_samples = len(signal_waveform)
  results = np.zeros((num_samples, num_trials))
  sum_spectrums_squared = 0
  for i in range(num_trials):
    noise_spectrum, noise_waveform = make_noise(num_samples, noise_level, 
                                                flat=flat)
    sum_spectrums_squared += np.abs(noise_spectrum)**2
    results[:, i] = signal_waveform + noise_waveform
  return results, np.sqrt(sum_spectrums_squared/num_trials)


def run_experiment(num_samples = 512, num_trials = 8192, flat=True):
  signal_spectrum = np.zeros(num_samples)
  signal_spectrum[4] = 0.5 # Just one frequency, 4 cycles per signal length.
  signal_spectrum[-4] = 0.5
  signal_spectrum *= num_samples # To mimic scaling by num_samples in forward FFT
  test_signal = make_signal(signal_spectrum)

  s_covariances = []
  n_covariances = []
  z_s = []
  s_s = [1, 2, 3, 4, 5]
  noise_level = 1
  for s in s_s:
    waveforms, noise_spectrum = make_experiment(s*test_signal,
                                                noise_level=noise_level,
                                                num_trials=num_trials,
                                                flat=flat)
    s_covariance, n_covariance = calculate_similarity(waveforms)
    s_covariances.append(s_covariance)
    n_covariances.append(n_covariance)

    z = (np.mean(s_covariance) - np.mean(n_covariance)) / np.sqrt(np.var(s_covariance) + np.var(n_covariance))
    z_s.append(z)
  return s_covariances, n_covariances, signal_spectrum, noise_spectrum, s_s, z_s


class ColoredNoise(absltest.TestCase):

  def test_noise_comparison(self) -> None:
    plot_file='AnalysisNoiseComparison.png'
    plot_dir: str = 'resultm'

    plt.figure(figsize=(18, 6))
    for i, flat in enumerate([True, False]):
      (s_covariances, n_covariances, signal_spectrum, noise_spectrum, 
      s_s, z_s) = run_experiment(flat=flat, num_trials=65536)

      plt.subplot(3, 2, i+1)
      plt.plot(s_s, [np.mean(s_covariance) for s_covariance in s_covariances], 'x', label='Simulation')
      plt.plot(s_s, [colored_theory_mean(signal_spectrum*s, np.sqrt(noise_spectrum)) for s in s_s], label='Theory')
      plt.plot(s_s, [np.mean(n_covariance) for n_covariance in n_covariances], 'x', label='Noise Simulation')
      plt.plot(s_s, [colored_theory_mean(signal_spectrum*0, np.sqrt(noise_spectrum)) for s in s_s], label='Noise Theory')
      plt.legend()
      plt.ylabel('Mean')
      flat_label = 'Flat Phase' if flat else 'Gaussian'
      plt.title(f'Colored Noise - {flat_label}');

      plt.subplot(3, 2, i+3)
      plt.plot(s_s, [np.var(s_covariance) for s_covariance in s_covariances], 'x', label='Simulation')
      plt.plot(s_s, [colored_theory_var(signal_spectrum*s, noise_spectrum) for s in s_s], label='Theory')
      plt.legend();
      plt.ylabel('Variance');

      plt.subplot(3, 2, i+5)
      plt.plot(s_s, z_s, 'x', label='Simulation')
      plt.plot(s_s, [colored_theory_dprime(signal_spectrum*s, noise_spectrum) for s in s_s], label='Theory')
      plt.legend();
      plt.xlabel('Signal Level')
      plt.title('d\'');

      # Save figure at each step before assertions, so we can see the plots 
      # even if the test fails.
      plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

      # Check simulated means for signal case
      np.testing.assert_allclose(
        [np.mean(s_covariance) for s_covariance in s_covariances],
        [colored_theory_mean(signal_spectrum*s, np.sqrt(noise_spectrum)) 
         for s in s_s],
         rtol=0.1)

      # Check simulated means for noise case
      np.testing.assert_allclose(
        [np.mean(n_covariance) for n_covariance in n_covariances],
        [colored_theory_mean(signal_spectrum*0, np.sqrt(noise_spectrum)) for s in s_s],
        atol=2.0)

      # Check simulated variances for the signal case
      np.testing.assert_allclose(
        [np.var(s_covariance) for s_covariance in s_covariances],
        [colored_theory_var(signal_spectrum*s, noise_spectrum) for s in s_s], 
        rtol=0.1)

      # Check simulated d-prime values
      np.testing.assert_allclose(
        z_s,
        [colored_theory_dprime(signal_spectrum*s, noise_spectrum) for s in s_s],
        rtol=0.1)

if __name__ == '__main__':
  absltest.main()