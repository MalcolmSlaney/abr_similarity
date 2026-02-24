import absl.app
import absl.flags

from datetime import datetime
import glob
import json
import absl
import numpy as np
from numpy.typing import ArrayLike, NDArray
import numpy.polynomial.polynomial as poly

import os
import pandas as pd
import random
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import zarr
from cftsdata import abr

from analyze import calculate_jackknife_covariance, randomize_phase


def read_experiment_data(path):
  """Load all the data for one animal experiment given its experiment name.
  One animal, one time, left/right

  Return:
    A panda data frame with all the data from that experiment.
  """
  load_options = {}

  if not os.path.exists(os.path.join(path, 'erp_metadata.csv')):
    raise ValueError(f'{path} does not contain ABR data')
  fh = abr.load(path)
  epochs = fh.get_epochs_filtered(**load_options)
  return epochs.sort_index()

  # for freq, freq_df in epochs.groupby('frequency'):
  #   yield freq, freq_df




def get_mouse_data(basedir, mouse_number, timepoint=None, left='*'):
  """Read all the ABR data from one experimental directory.  A directory
  seems to containe one animal, one ear, and one day in time.  The resulting
  data frame has multiple frequencies, levels, signal polarities, and the
  resulting ERP data.

  Returns:
    A dataframe
  """
  if timepoint is None:
    timepoint = '*'
  exps = glob.glob(os.path.join(basedir, f'Mouse{mouse_number}_timepoint{timepoint}_{left} *'))
  if len(exps) > 1:
    raise ValueError(f'Got more than one experiment: {[d.replace(basedir + "/", "") for d in exps]}')
  if len(exps) == 0:
    raise IOError(f'No experiments found for mouse {mouse_number}')
  return read_experiment_data(exps[0])



def get_unique_levels(freq_df):
  levels = []
  for f, l, p, t in freq_df.index:
    levels.append(l)
  return sorted(list(set(levels)))

def get_unique_freqs(freq_df):
  freqs = []
  for f, l, p, t in freq_df.index:
    freqs.append(f)
  return sorted(list(set(freqs)))

def get_unique_polarities(freq_df):
  polarities = []
  for f, l, p, t in freq_df.index:
    polarities.append(p)
  return list(set(polarities))

def get_summary(df):
  return get_unique_levels(df), get_unique_freqs(df), get_unique_polarities(df)



def get_one_exp_type(one_exp_data, freq, level, polarity='both'):
  """From a panda dataframe containing all the data from one experiment,
  extract the ERPs for one frequency, sound level, and one or
  more polarities.

  If polarity is both, then add positive and negative going clicks to
  get an average response without the electrical signal.

  Returns:
    nd.array, of size num_trials x num_times
  """
  if polarity == 'both':
    pos = one_exp_data.loc[(freq, level, 1)].to_numpy()
    neg = one_exp_data.loc[(freq, level, -1)].to_numpy()
    num = min(pos.shape[0], neg.shape[0])
    result = pos[:num, :] + neg[:num, :]
  else:
    result = one_exp_data.loc[(freq, level, polarity)].to_numpy()
  assert result.shape[1] == 243
  return result



def calculate_similarity(
    signal_waveforms: NDArray,
    noise_waveforms: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
  s_covariances = calculate_jackknife_covariance(signal_waveforms)
  if noise_waveforms is None:
    noise_waveforms = randomize_phase(signal_waveforms)
  else:
    noise_waveforms = noise_waveforms 
  n_covariances = calculate_jackknife_covariance(noise_waveforms)
  return s_covariances, n_covariances

def dprime_from_distributions(signal_distribution: NDArray,
                              noise_distribution: Optional[NDArray] = None) -> float:
  return float((np.mean(signal_distribution) - np.mean(noise_distribution)) /
               np.sqrt(0.5 * (np.var(signal_distribution) + 
                              np.var(noise_distribution))))

def calculate_dprime(signal_waveforms: NDArray,
    noise_waveforms: Optional[NDArray] = None) -> float:
  s_covariances, n_covariances = calculate_similarity(signal_waveforms, noise_waveforms)
  return dprime_from_distributions(s_covariances, n_covariances)
  

def calculate_dprime_stack(exp_df, freq, plot_stack=False):
  levels = sorted(get_unique_levels(exp_df))
  if not levels:
    raise ValueError('No levels found')
  freqs = sorted(get_unique_freqs(exp_df))
  if freq not in freqs:
    raise ValueError(f'Frequency {freq} must be in {freqs}.')
  noisy_data = get_one_exp_type(exp_df, freq, levels[0], 'both').T.copy()
  np.random.shuffle(noisy_data)

  if plot_stack:
    plt.figure(figsize=(8, 12))
    data = get_one_exp_type(exp_df, freq, levels[-1], 'both').T
    label_max = np.max(np.abs(np.mean(data, axis=1)))
  dprimes = []
  dprimes_wo_noise = []
  for i, level in enumerate(levels):
    data = get_one_exp_type(exp_df, freq, level, 'both').T
    if plot_stack:
      plt.subplot(len(levels), 1, i + 1)
      plt.plot(np.mean(data, axis=1), label=f'{level}dB')
      plt.plot(np.mean(noisy_data, axis=1), label='random')
      plt.ylim(-label_max, label_max)
      plt.xticks([])
      plt.ylabel(level)
    dprimes.append(calculate_dprime(data, noisy_data))
    dprimes_wo_noise.append(calculate_dprime(data, None))
  return levels, dprimes, dprimes_wo_noise


class DPrimeQuadratic(object):
  def __init__(self, levels: ArrayLike, dprimes: ArrayLike, order: int = 2, plot=False):
    # self.poly_coeffs = np.polyfit(levels, dprimes, order)
    # self.quad_function = np.poly1d(self.poly_coeffs)
  
    self.dp_poly = poly.Polynomial.fit(levels, dprimes, deg=[2,], domain=[-100, 100])
    if plot:
      plt.plot(levels, dprimes, 'o', label='Original Data')
      plt.plot(levels, self.dp_poly(levels), '-', label='Quadratic Fit')
      plt.legend()
      plt.title('Quadratic Fit to d\' vs. Level')
      plt.xlabel('Level (dB)')
      plt.ylabel('d\'')
      plt.grid(True)

  def compute(self, levels: ArrayLike) -> ArrayLike:
    return self.dp_poly(levels)


def get_level_for_dprime(levels, dprimes, target_dprime):
  """Estimates the sound level required to achieve a given d-prime value.

  Args:
    levels: A list or array of sound levels.
    dprimes: A list or array of corresponding d-prime values.
    target_dprime: The desired d-prime value.

  Returns:
    The estimated sound level (float) corresponding to the target d-prime.
    Returns None if no real solution is found within the range of levels.
  """
  # Fit a quadratic polynomial to the levels and dprimes
  poly_coeffs = np.polyfit(levels, dprimes, 2)

  # Create a new polynomial for which we want to find the root:
  # a*x^2 + b*x + c - target_dprime = 0
  a, b, c = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2] - target_dprime

  # Solve the quadratic equation ax^2 + bx + c = 0 for x (level)
  # Using the quadratic formula: x = (-b +- sqrt(b^2 - 4ac)) / 2a
  discriminant = b**2 - 4*a*c

  if discriminant < 0:
    # No real roots
    return None
  elif discriminant == 0:
    # One real root
    root = -b / (2*a)
    roots = [root]
  else:
    # Two real roots
    root1 = (-b + np.sqrt(discriminant)) / (2*a)
    root2 = (-b - np.sqrt(discriminant)) / (2*a)
    roots = [root1, root2]

  # Filter for roots that are within the range of the input levels
  min_level, max_level = min(levels), max(levels)
  valid_roots = [r for r in roots if min_level <= r <= max_level and np.isreal(r)]

  if not valid_roots:
    return None
  
  # Return the root closest to the center of the level range if multiple valid roots
  # or simply the first valid root found
  if len(valid_roots) > 1:
    # If the d-prime curve is U-shaped, we generally want the higher level
    # or the one that makes sense in the context of increasing d-prime with level
    # For ABR, d-prime increases with level, so we want the larger root
    if a > 0: # Parabola opens upwards, higher d' comes from higher level
        return max(valid_roots)
    else: # Parabola opens downwards, higher d' comes from lower level
        return min(valid_roots)
  else:
    return valid_roots[0]


def compare_dprime_to_thresholds(threshold_df, basedir):
  matched_levels = []
  matched_dprimes = []
  last_mouse_key = None
  # for index, row in manual_threshs.iterrows():
  for index, row in threshold_df.iterrows():
    mouse_id = row['id']
    timepoint = row['timepoint']
    ear = row['ear']
    frequency = row['frequency']
    if 'manual threshold' in row:
      manual_threshold = row['manual threshold']
    elif 'threshold' in row:
      manual_threshold = row['threshold']
    else:
      raise ValueError('Threshold column not found in the dataframe.')

    if last_mouse_key != (mouse_id, timepoint, ear):
      try:
        good_df = get_mouse_data(basedir, mouse_id, timepoint, ear)
      except:
        continue
      last_mouse_key = (mouse_id, timepoint, ear)
    print(f"{mouse_id}: Timepoint: {timepoint}, Ear: {ear}, Frequency: {frequency}, Manual Threshold: {manual_threshold}")
    levels, freqs, polarities = get_summary(good_df)
    # high_data = get_one_exp_type(good_df, frequency, 85.0, 'both').T

    # for freq in freqs:
    if True:
      freq = frequency
      levels, dprimes, dprimes_without_noise = calculate_dprime_stack(good_df, freq, plot_stack=False)
      # plt.figure()
      dpq = DPrimeQuadratic(levels, dprimes, plot=False)
      if np.isfinite(dpq.compute(manual_threshold)):
        matched_levels.append(manual_threshold)
        matched_dprimes.append(dpq.compute(manual_threshold))
      # break
  return matched_levels, matched_dprimes


def get_threshold_data(basedir: str, csv_filename: str = 'Manual Thresholds.csv') -> Optional[pd.DataFrame]:
  csv_path = os.path.join(basedir, csv_filename)
  try:
    manual_df = pd.read_csv(csv_path)
    # display(manual_df.head(20))
    return manual_df
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: The file '{csv_path}' was not found.")


from sklearn.linear_model import LinearRegression

def fit_linear_regression(levels: ArrayLike, dprimes: ArrayLike) -> Tuple[float, float]:
  """Fit a linear regression model to the data and return the slope and intercept."""
  # Reshape matched_levels for sklearn (it expects a 2D array)
  X = np.array(levels).reshape(-1, 1)
  y = np.array(dprimes)

  # Create and fit the linear regression model
  model = LinearRegression()
  model.fit(X, y)
  return model.coef_[0], model.intercept_

absl.flags.DEFINE_string('basedir', '../ABRPrestoData', 'Base directory containing the ABR data and threshold CSV files.')

FLAGS = absl.flags.FLAGS

def main(argv):
  del argv  # Unused

  # First compare the covariance-based d-prime estimates to the ABRPresto thresholds

  cache_filename = 'Results/ABRPrestoThresholdData.npz'
  if os.path.exists(cache_filename):
    data = np.load(cache_filename)
    matched_abrpresto_dprimes = data['matched_abrpresto_dprimes']
    matched_abrpresto_levels = data['matched_abrpresto_levels']
    abrpresto_slope = data['abrpresto_slope']
    abrpresto_intercept = data['abrpresto_intercept']
  else:
    abrpresto_df = get_threshold_data(FLAGS.basedir, 'ABRpresto thresholds 10-29-24.csv')
    print(abrpresto_df.head())
    matched_abrpresto_levels, matched_abrpresto_dprimes = compare_dprime_to_thresholds(abrpresto_df, basedir=FLAGS.basedir)
    abrpresto_slope, abrpresto_intercept = fit_linear_regression(matched_abrpresto_levels, matched_abrpresto_dprimes)
    # Write the dictionary to a cache file
    np.savez(cache_filename,
             matched_abrpresto_levels=matched_abrpresto_levels,
             matched_abrpresto_dprimes=matched_abrpresto_dprimes,
             abrpresto_slope=abrpresto_slope,
             abrpresto_intercept=abrpresto_intercept,
             datetime=str(datetime.now()),
    )

  plt.figure()
  plt.plot(matched_abrpresto_levels, matched_abrpresto_dprimes, 'x', alpha=0.1)
  plt.plot(matched_abrpresto_levels, [abrpresto_slope * x + abrpresto_intercept for x in matched_abrpresto_levels], label='ABRPresto Threshold Fit')
  plt.xlabel('Level (dB)')
  plt.ylabel('d\'')
  plt.title(f'Comparison between ABR Threshold and Covariance')
  plt.text(50, abrpresto_slope * 50 + abrpresto_intercept + 0.03, 
           f'Slope: {abrpresto_slope*10:.2f}/10dB, Intercept: {abrpresto_intercept:.2f}', 
           rotation=np.arctan(abrpresto_slope) * 180 / np.pi, fontsize=10,
           rotation_mode='anchor', transform_rotates_text=True)
  plt.savefig('Results/ThresholdComparisonABRPresto.png')

  # Now compare the covariance-based d-prime estimates to the manual thresholds
  cache_filename = 'Results/ABRPrestoManualData.npz'
  if os.path.exists(cache_filename):
    data = np.load(cache_filename)
    matched_manual_levels = data['matched_manual_levels']
    matched_manual_dprimes = data['matched_manual_dprimes']
    matched_slope = data['manual_slope']
    matched_intercept = data['manual_intercept']
  else:
    manual_df = get_threshold_data(FLAGS.basedir, 'Manual Thresholds.csv')
    print(manual_df.head())
    
    matched_manual_levels, matched_manual_dprimes = compare_dprime_to_thresholds(manual_df, basedir=FLAGS.basedir)
    matched_slope, matched_intercept = fit_linear_regression(matched_manual_levels, matched_manual_dprimes)
    np.savez(cache_filename,
             matched_manual_levels=matched_manual_levels,
             matched_manual_dprimes=matched_manual_dprimes,
             manual_slope=matched_slope,
             manual_intercept=matched_intercept,
             datetime=str(datetime.now()),
    )


  plt.figure()
  plt.plot(matched_manual_levels, matched_manual_dprimes, 'x', alpha=0.1)
  plt.plot(matched_manual_levels, [matched_slope * x + matched_intercept for x in matched_manual_levels], label='Manual Threshold Fit')
  plt.xlabel('Level (dB)')
  plt.ylabel('d\'')
  plt.title(f'Comparison between Manual Threshold and Covariance')
  plt.text(50, matched_slope * 50 + matched_intercept + 0.03, 
           f'Slope: {matched_slope*10:.2f}/10dB, Intercept: {matched_intercept:.2f}', 
           rotation=np.arctan(matched_slope) * 180 / np.pi, fontsize=10,
           rotation_mode='anchor', transform_rotates_text=True)
  plt.savefig('Results/ThresholdComparisonManual.png')

if __name__ == '__main__':
  absl.app.run(main)