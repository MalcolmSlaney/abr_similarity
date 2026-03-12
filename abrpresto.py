import absl.app
import absl.flags

import os
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple, Union

from datetime import datetime
import glob
import json
import absl
import numpy as np
from numpy.typing import ArrayLike, NDArray
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import zarr
from cftsdata import abr
from sklearn.linear_model import LinearRegression

from analyze import calculate_jackknife_covariance, randomize_phase
from abrpresto_utils import FitPowerCurve, FitQuadraticMonomialCurve, FitSigmoidCurve


################## Access the published ABRPresto data  ##################

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

################## Calculate Similarity and d' ##################


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
  

def calculate_dprime_stack(exp_df: pd.DataFrame, freq: float, plot_stack: bool = False):
  levels = sorted(get_unique_levels(exp_df), reverse=True)  # Want biggest down to smallest
  if not levels:
    raise ValueError('No levels found')
  freqs = sorted(get_unique_freqs(exp_df))
  if freq not in freqs:
    raise ValueError(f'Frequency {freq} must be in {freqs}.')
  noisy_data = get_one_exp_type(exp_df, freq, levels[-1], 'both').T.copy()
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



from dataclasses import dataclass, field

trial_dprimes = [.001, .0025, 0.05, .01, 0.025, 0.5, 0.1, 0.25, 0.5, 1.0]
@dataclass
class ABRSummary(object):
  manual_threshold: float = 0
  abrpresto_threshold: float = 0
  dprime_at_manual_threshold: float = 0
  dprime_at_abrpresto_threshold: float = 0
  dprime_thresholds: List[float] = field(default_factory=list)  # One per trial_dprimes


def evaluate_thresholds(
    summaries: Dict[Tuple, ABRSummary], 
    error_threshold: float = 10) -> Tuple[List[float], List[float]]:
  accuracies = np.zeros((len(trial_dprimes),))
  for k, summary in summaries.items():
    for i, d in enumerate(trial_dprimes):
      estimated_level = summary.dprime_thresholds[i]
      if np.isfinite(estimated_level):
        if abs(estimated_level - summary.manual_threshold) <= error_threshold:
          accuracies[i] += 1
  accuracies /= len(summaries)
  return trial_dprimes, accuracies

##################  Summarize all the ABRPresto data  ##################

def summarize_all_data(manual_df: pd, 
                       abr_presto_df: pd, 
                       basedir: dir, 
                       power_fit: bool = True) -> Dict[Tuple, ABRSummary]:
  last_mouse_key = None
  summaries = {}
  for index, row in manual_df.iterrows():
    mouse_id = row['id']
    timepoint = row['timepoint']
    ear = row['ear']
    frequency = row['frequency']
    if 'manual threshold' in row:
      manual_threshold = row['manual threshold']
    else:
      continue

    # We only want to get the mouse data off disk once, so keep track if the 
    # same data is being requested again and only read from disk if it's a new 
    # mouse/timepoint/ear combination.  This is because the threshold data has 
    # multiple rows for each mouse/timepoint/ear combination, one for each 
    # frequency.   
    if last_mouse_key != (mouse_id, timepoint, ear):
      try:
        good_df = get_mouse_data(basedir, mouse_id, timepoint, ear)
      except:
        continue
      last_mouse_key = (mouse_id, timepoint, ear)
    print(f'Computing {mouse_id}: Timepoint: {timepoint}, Ear: {ear}, Frequency: {frequency}, Manual Threshold: {manual_threshold}')
    levels, dprimes, _ = calculate_dprime_stack(good_df, frequency, 
                                                plot_stack=False)
    print(f'Levels: {levels}, D-primes: {dprimes}')
    if power_fit:
      dpq = FitPowerCurve(levels, dprimes, plot=False)
    else:
      dpq = FitQuadraticMonomialCurve(levels, dprimes, plot=False)
    abr_summary = ABRSummary()
    abr_summary.manual_threshold = manual_threshold
    abr_summary.abrpresto_threshold = abr_presto_df.loc[
      (abr_presto_df['id'] == mouse_id) &
      (abr_presto_df['timepoint'] == timepoint) &
      (abr_presto_df['ear'] == ear) &
      (abr_presto_df['frequency'] == frequency),
      'threshold'].values[0]
    abr_summary.dprime_at_manual_threshold = dpq.compute(manual_threshold)
    abr_summary.dprime_at_abrpresto_threshold = dpq.compute(abr_summary.abrpresto_threshold)
    abr_summary.dprime_thresholds = [dpq.inverse_compute(d) for d in trial_dprimes]
    summaries[(mouse_id, timepoint, ear, frequency)] = abr_summary
  return summaries


def compare_dprime_to_thresholds(threshold_df, basedir: str, power_fit: bool = True):
  """For each threshold in a Panda dataframe (ABRPresto or manual) create a pair
  of lists comparing the threshold data with the abr similarity d-prime 
  estimates.  The two returned lists contain the threshold (either ABRPresto or 
  Manual) and the corresponding d-prime estimate for that experiment.  

  The d-prime estimate is obtained by fitting a curve to the d-prime vs. level 
  data and then interpolating to find the d-prime value at the threshold level.
  """
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
      manual_threshold = row['threshold']    # From ABRPresto threshold data
    else:
      raise ValueError('Threshold column not found in the dataframe.')

    # We only want to get the mouse data off disk once, so keep track if the 
    # same data is being requested again and only read from disk if it's a new 
    # mouse/timepoint/ear combination.  This is because the threshold data has 
    # multiple rows for each mouse/timepoint/ear combination, one for each 
    # frequency.   
    if last_mouse_key != (mouse_id, timepoint, ear):
      try:
        good_df = get_mouse_data(basedir, mouse_id, timepoint, ear)
      except:
        continue
      last_mouse_key = (mouse_id, timepoint, ear)
    print(f"{mouse_id}: Timepoint: {timepoint}, Ear: {ear}, Frequency: {frequency}, Manual Threshold: {manual_threshold}")
    levels, freqs, polarities = get_summary(good_df)

    freq = frequency
    levels, dprimes, dprimes_without_noise = calculate_dprime_stack(good_df, freq, plot_stack=False)
    if power_fit:
      dpq = FitPowerCurve(levels, dprimes, plot=False)
    else:
      dpq = FitQuadraticMonomialCurve(levels, dprimes, plot=False)

    if np.isfinite(dpq.compute(manual_threshold)):
      matched_levels.append(manual_threshold)
      matched_dprimes.append(dpq.compute(manual_threshold))
  return matched_levels, matched_dprimes


def show_dprime_interpolation(basedir: str, manual_df, mouse_id: int = 140, 
                             timepoint: int = 0, channel: str = 'left', 
                             plot_filename: str = None):
  good_df = get_mouse_data(basedir, mouse_id, timepoint, channel)
  freqs = sorted(get_unique_freqs(good_df))
  plt.figure(figsize=(10, 10))
  for i, freq in enumerate(freqs):
    print(i, freq)
    plt.subplot(3, 3, i+1)
    levels, dprimes, dprimes_without_noise = calculate_dprime_stack(good_df, freq)
    dpq = FitPowerCurve(levels, dprimes, plot=True)
    manual_threshold_value = manual_df.loc[
      (manual_df['id'] == mouse_id) &
      (manual_df['timepoint'] == 0) &
      (manual_df['frequency'] == freq),
      'manual threshold'
    ].values[0]

    plt.title(f"{freq}Hz - {manual_threshold_value}dB")
    plt.axvline(float(dpq.fitted_breakpoint), ls=':');
    if plot_filename:
      plt.savefig(plot_filename)


def get_threshold_data(basedir: str, csv_filename: str = 'Manual Thresholds.csv') -> Optional[pd.DataFrame]:
  csv_path = os.path.join(basedir, csv_filename)
  try:
    manual_df = pd.read_csv(csv_path)
    # display(manual_df.head(20))
    return manual_df
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: The file '{csv_path}' was not found.")



def fit_linear_regression(levels: ArrayLike, dprimes: ArrayLike) -> Tuple[float, float]:
  """Fit a linear regression model to the data and return the slope and intercept."""
  # Reshape matched_levels for sklearn (it expects a 2D array)
  X = np.array(levels).reshape(-1, 1)
  y = np.array(dprimes)

  # Create the boolean mask for finite values in all the data.
  mask = np.logical_and(np.isfinite(y), np.isfinite(X.flatten()))

  # Apply the mask to both arrays
  X = X[mask]
  y = y[mask]

  # Create and fit the linear regression model
  model = LinearRegression()
  model.fit(X, y)
  return model.coef_[0], model.intercept_

absl.flags.DEFINE_string('basedir', '../ABRPrestoData', 'Base directory containing the ABR data and threshold CSV files.')


def compute_pearson_correlation(levels: ArrayLike, dprimes: ArrayLike) -> float:
  """Compute the Pearson correlation coefficient between levels and d-primes."""
  return np.corrcoef(levels, dprimes)[0, 1]

FLAGS = absl.flags.FLAGS


def clean_key(key):
    # Example transformation: replace "item" with "object"
    return " ".join([str(k) for k in key])


def restore_key(cleaned_key):
  return tuple(cleaned_key.split(' '))

##################  Summarize all the ABRPresto data  ##################

def main(argv):
  del argv  # Unused
  global trial_dprimes

  manual_df = get_threshold_data(FLAGS.basedir, 'Manual Thresholds.csv')
  abr_presto_df = get_threshold_data(FLAGS.basedir, 'ABRpresto thresholds 10-29-24.csv')

  cache_filename = 'Results/ABRPrestoSummary.json'
  if os.path.exists(cache_filename):
    with open(cache_filename, 'r') as f:
      all_data = json.load(f)
      summaries = all_data['summaries']
      summaries = {restore_key(key): ABRSummary(**value) for key, value in summaries.items()}
      trial_dprimes = all_data['dprimes']
    print(f'Loaded cached summaries for {len(summaries)} experiments from {cache_filename}.')
  else:
    summaries = summarize_all_data(manual_df, abr_presto_df, FLAGS.basedir)
    with open(cache_filename, 'w') as f:
      new_summaries = {clean_key(key): value.__dict__ for key, value in summaries.items()}
      json.dump({'summaries': new_summaries,
                 'dprimes': trial_dprimes}, 
                f, indent=2)
    print(f'Cached summaries for {len(summaries)} experiments to {cache_filename}.')
  for k, summary in summaries.items():
    print(f'{k}: Manual Threshold: {summary.manual_threshold}, ABRPresto Threshold: {summary.abrpresto_threshold}, D-prime Thresholds: {summary.dprime_thresholds}')  
    break

  matched_abrpresto_levels = np.array([abr_summary.abrpresto_threshold 
                                       for abr_summary in summaries.values()])
  matched_abrpresto_dprimes = np.array([abr_summary.dprime_at_abrpresto_threshold 
                                        for abr_summary in summaries.values()])
  mask = np.logical_and(np.isfinite(matched_abrpresto_levels), 
                        np.isfinite(matched_abrpresto_dprimes))
  matched_abrpresto_levels = matched_abrpresto_levels[mask]
  matched_abrpresto_dprimes = matched_abrpresto_dprimes[mask]

  abrpresto_correlation = compute_pearson_correlation(matched_abrpresto_levels, 
                                                      matched_abrpresto_dprimes)
  print(f'ABRPresto threshold correlation r={abrpresto_correlation:.2f}')

  abrpresto_slope, abrpresto_intercept = fit_linear_regression(matched_abrpresto_levels, matched_abrpresto_dprimes)

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
  plt.ylim(-0.25, 1.5)
  plt.savefig('Results/ThresholdComparisonABRPresto.png')

  matched_manual_levels = np.array([abr_summary.manual_threshold 
                                  for abr_summary in summaries.values()])
  matched_manual_dprimes = np.array([abr_summary.dprime_at_manual_threshold 
                                   for abr_summary in summaries.values()])
  mask = np.logical_and(np.isfinite(matched_manual_levels), 
                        np.isfinite(matched_manual_dprimes))
  matched_manual_levels = matched_manual_levels[mask]
  matched_manual_dprimes = matched_manual_dprimes[mask]
  manual_correlation = compute_pearson_correlation(matched_manual_levels, matched_manual_dprimes)
  print(f'Manual threshold correlation r={manual_correlation:.2f}') 
  matched_slope, matched_intercept = fit_linear_regression(matched_manual_levels, matched_manual_dprimes)

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
  plt.ylim(-0.25, 1.5)
  plt.savefig('Results/ThresholdComparisonManual.png')

  trial_dprimes, accuracies = evaluate_thresholds(summaries, error_threshold=10)
  print('Covariance accuracies on ABRPresto data versus d\':', accuracies)
  plt.clf()
  plt.semilogx(trial_dprimes, accuracies, 'o-')
  plt.xlabel('D-prime Threshold')
  plt.ylabel('Accuracy within 10dB of Manual Threshold')
  plt.title('Accuracy of D-prime Thresholds Compared to Manual Thresholds')
  plt.savefig('Results/ABRPrestoThresholdAccuracy.png')

  show_dprime_interpolation(FLAGS.basedir, manual_df, mouse_id=242, 
                            timepoint=0, channel='left',
                            plot_filename='Results/ThresholdInterpolationExample.png')
  return

if __name__ == '__main__':
  absl.app.run(main)