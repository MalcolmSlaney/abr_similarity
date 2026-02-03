from absl import app
from absl import flags
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.typing import NDArray, ArrayLike
from typing import List, Tuple

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
from sklearn.linear_model import Ridge, LinearRegression

flags.DEFINE_integer('seed', 42, 'Initial seed for data generation.')
flags.DEFINE_integer('num_experiments', 1000, 'How many experiments to run.')
flags.DEFINE_integer('num_jackknife_experiments', 100, 'How many jackknife experiments to run.')
flags.DEFINE_string('cache_dir', '.', 'Directory to cache simulation results.')

FLAGS = flags.FLAGS


# Constants for these Trials

signal_levels = 10**np.arange(-3, 0.51, .25)
default_noise_level = 1.2

correlation_trial_count_list = [1, 2, 10]
correlation_trial_count_list = np.asarray(sorted(list(set([int(f) for f in 10**np.arange(0, 4.1, 0.25)]))))

spc_d_threshold = 1 # What d' do we want for threshold?
spf_d_threshold = 1 # What d' do we want for threshold?
spp_d_threshold = 1 # What d' do we want for threshold?
spjk_d_threshold = 1 # What d' do we want for threshold?
spml_d_threshold = 1 # Threshold for multi-look tests

# print('Correlation Trial Count List:', correlation_trial_count_list)


def matched_filter_correlate(w: NDArray) -> Tuple[NDArray, NDArray]:
  """Implement the matched filter correlation.  Correlate each
  trial the average over the experiment. Compute the mean and variance.
  """
  N, K = w.shape
  means = np.zeros(K)
  vars = np.zeros(K)
  model = np.mean(w, axis=0, keepdims=True)
  correlations = model*w
  return np.mean(correlations), np.var(correlations)

# matched_filter_correlate(np.random.randn(3, 4))


def full_correlate(w: NDArray, max_k=20) -> Tuple[NDArray, NDArray]:
  """Implement the full trial-by-trial correlation.  Correlate each
  trial with each other. Compute the mean and variance.

  This routine does this calculation one experiment at a time, and limits
  the calculations to no more than max_k experiments to save time.

  Args:
    w is a single point of the waveform data of size N trials x K experiments.
  """
  N, K = w.shape # Num trials x num experiments
  K = min(K, max_k)
  means = np.zeros(K)
  vars = np.zeros(K)
  for i in range(K):
    correlation = w[:, i:i+1] * w[:, i:i+1].T
    assert correlation.shape == (N, N)
    means[i] = np.mean(correlation)
    vars[i] = np.var(correlation)
  return np.mean(means), np.mean(vars)

# full_correlate(np.random.randn(3, 4))

def jackknife_correlate(data):
    """Compute the correlation of the data with a model that does *not* include
    the trial data.

    Args:
        data: measurements of size num_trials x num_experiments
    Returns
        Array of size num_trials x num_experiments
    """
    results = np.zeros(data.shape)
    for i in range(data.shape[0]):
        other_data = np.concatenate((data[:i, :], data[i+1:, :]), axis=0)
        model = np.mean(other_data, axis=0)
        results[i, :] = data[i, :] * model
    return results

def simulate_point_process(
    n: float = 1.2, num_experiments: int = 1000,
    signal_levels: ArrayLike = 10**np.arange(-3, 0.51, .25),
    correlation_trial_count_list: ArrayLike = np.asarray(sorted(list(set([int(f) for f in 10**np.arange(0, 4.1, 0.25)])))),
    jackknife: bool = False):
    """Generate simulated single-point ABR data and measure it's correlation and power metrics.
    Use the same data for both approaches to maximize commonality.

    Args:
        n: the amplitude of the noise (standard deviation of a Gaussian)
        num_experiments: How many experiments, each with multiple trials, to run
        jackknife: Whether to use jackkniving to build each model from data we don't use for testing

    Returns:
        spc_dprimes: Estimated d' for the single-point correlation measure
        spp_dprimes: Estimated d' for the single-point power measure
        spc_mean_noise: The mean, per experiment, of the correlation measure of the noise (no signal)
        spc_mean_signal: The mean, per experiment, of the correlation measure of the signal
        spc_var_noise: The variance, across experiments, of the correlation measure of the noise (no signal)
        spc_var_signal: The variance, across experiments, of the correlation measure of the signal
        spp_mean_noise: The mean, per experiment, of the power measure of the noise (no signal)
        spp_mean_signal: The mean, per experiment, of the power measure of the signal
        spp_var_noise: The variance, across experiments, of the power measure of the noise (no signal)
        spp_var_signal: The variance, across experiments, of the power measure of the signal
    """
    print(f'Running the simulation with {num_experiments} experiments, '
          f'{len(signal_levels)} signal levels, {len(correlation_trial_count_list)} '
          f'different trial counts, jackknife={jackknife}')
    spc_dprimes = np.zeros((len(signal_levels), len(correlation_trial_count_list)))  # Single Point Correlation d'
    spp_dprimes = spc_dprimes.copy()  # Single Point Power d'
    spf_dprimes = spc_dprimes.copy()  # Full covariacne d'

    spp_mean_signal = spp_dprimes.copy()
    spp_mean_noise = spp_dprimes.copy()
    spp_var_signal = spp_dprimes.copy()
    spp_var_noise = spp_dprimes.copy()

    spc_mean_signal = spp_dprimes.copy()
    spc_mean_noise = spp_dprimes.copy()
    spc_var_signal = spp_dprimes.copy()
    spc_var_noise = spp_dprimes.copy()

    spf_mean_noise = spp_dprimes.copy()
    spf_var_noise = spp_dprimes.copy()
    spf_mean_signal = spp_dprimes.copy()
    spf_var_signal = spp_dprimes.copy()

    np.random.seed(FLAGS.seed)

    # Estimate measures over different signal levels and number of trials.
    for i, s in enumerate(signal_levels):
      for j, trial_count in enumerate(correlation_trial_count_list):
        measurements = n*np.random.randn(trial_count, num_experiments) + s  # The measured ABR signals
        # Must get new noise for the no-signal measurement.
        noise = n*np.random.randn(trial_count, num_experiments)

        # Now compute the correlations against the model, average across trials.
        if jackknife == False:
            signal_model = np.mean(measurements, axis=0, keepdims=True)
            noise_model = np.mean(noise, axis=0, keepdims=True)
            noise_correlations = noise * noise_model
            signal_correlations = measurements * signal_model
        else:
            noise_correlations = jackknife_correlate(noise)
            signal_correlations = jackknife_correlate(measurements)
        assert signal_correlations.shape == noise_correlations.shape == (trial_count, num_experiments)

        # Now compute RMS over each group of looks.
        # noise_distances = np.sqrt(np.mean(noise_correlations ** 2, axis=0))
        # signal_distances = np.sqrt(np.mean(signal_correlations ** 2, axis=0))
        # assert signal_distances.shape == (num_experiments,)

        # Compute d' across all the trials and all the experiments.
        spc_mean_noise[i, j] = np.mean(noise_correlations)
        spc_mean_signal[i, j] = np.mean(signal_correlations)
        spc_var_noise[i, j] = np.var(noise_correlations)
        spc_var_signal[i, j] = np.var(signal_correlations)
        spc_dprimes[i, j] = (spc_mean_signal[i, j] - spc_mean_noise[i, j])/(
                    np.sqrt((spc_var_signal[i, j] + spc_var_noise[i, j])/2))

        # Now measure power stats for the same data.
        ave_signal = np.mean(measurements, axis=0)**2
        ave_noise = np.mean(noise, axis=0)**2
        spp_mean_noise[i, j] = np.mean(ave_noise)
        spp_mean_signal[i, j] = np.mean(ave_signal)
        spp_var_noise[i, j] = np.var(ave_noise)
        spp_var_signal[i, j] = np.var(ave_signal)
        spp_dprimes[i, j] = (spp_mean_signal[i, j] - spp_mean_noise[i, j])/(
                    np.sqrt((spp_var_signal[i, j] + spp_var_noise[i, j])/2))

        # Now do the full correlation calculations
        full_noise_mean, full_noise_var = full_correlate(noise)
        full_signal_mean, full_signal_var = full_correlate(measurements)
        spf_mean_noise[i, j] = full_noise_mean
        spf_var_noise[i, j] = full_noise_var
        spf_mean_signal[i, j] = full_signal_mean
        spf_var_signal[i, j] = full_signal_var
        spf_dprimes[i, j] = (spf_mean_signal[i, j] - spf_mean_noise[i, j])/(
                    np.sqrt((spf_var_signal[i, j] + spf_var_noise[i, j])/2))
        # print('Full correlate:', i, j, full_correlate(measurements))

    return (
      spp_dprimes, spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
      spc_dprimes, spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
      spf_dprimes, spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal, 
      signal_levels)


def get_simulation_data(cache_dir: str = '.', jackknife: bool = False,
                        num_experiments=20) -> Tuple:
  """Run the simulations and plot the results.
  """
  cache_filename = f'covariance_cache-{default_noise_level}-jackknife_{jackknife}.npz'
  cache_filename = os.path.join(cache_dir, cache_filename)
  if os.path.exists(cache_filename):
    data = np.load(cache_filename)
    spp_dprimes = data['spp_dprimes']
    spp_mean_noise = data['spp_mean_noise']
    spp_mean_signal = data['spp_mean_signal']
    spp_var_noise = data['spp_var_noise']
    spp_var_signal = data['spp_var_signal']

    spc_dprimes = data['spc_dprimes']
    spc_mean_noise = data['spc_mean_noise']
    spc_mean_signal = data['spc_mean_signal']
    spc_var_noise = data['spc_var_noise']
    spc_var_signal = data['spc_var_signal']

    spf_dprimes = data['spf_dprimes']
    spf_mean_noise = data['spf_mean_noise']
    spf_mean_signal = data['spf_mean_signal']
    spf_var_noise = data['spf_var_noise']
    spf_var_signal = data['spf_var_signal']

    signal_levels = data['signal_levels']
    print(f'Loaded simulation results from {cache_filename}.')
  else:
    # Just do the highest signal level, and the highest trial count,
    # to make sure the code works.

    # spc is single point correlation
    # spf is single point full correlation
    # spp is single point power metric
    # spjk is single point correlation via jackknife

    (spp_dprimes, spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
     spc_dprimes, spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
     spf_dprimes, spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal, 
     signal_levels) = simulate_point_process(
      n=default_noise_level, 
      num_experiments=num_experiments,
      jackknife=jackknife,
    )
    np.savez(cache_filename,
             spp_dprimes=spp_dprimes,
             spp_mean_noise=spp_mean_noise,
             spp_mean_signal=spp_mean_signal,
             spp_var_noise=spp_var_noise,
             spp_var_signal=spp_var_signal,

             spc_dprimes=spc_dprimes,
             spc_mean_noise=spc_mean_noise,
             spc_mean_signal=spc_mean_signal,
             spc_var_noise=spc_var_noise,
             spc_var_signal=spc_var_signal,

             spf_dprimes=spf_dprimes,
             spf_mean_noise=spf_mean_noise,
             spf_mean_signal=spf_mean_signal,
             spf_var_noise=spf_var_noise,
             spf_var_signal=spf_var_signal,

             signal_levels=signal_levels,
             jackknife=jackknife,
             datetime=str(datetime.now()),
             )
    print('Saving simulation results to', cache_filename)
  return (
    spp_dprimes, spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
    spc_dprimes, spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
    spf_dprimes, spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal, 
    signal_levels)



def compare_full_partial_correlation(plot_dir: str = '.', 
                                     figsize=(6.4, 4.8),
                                     plot_file='FullVsPartialCovariance.png'):
  """Compare the full covariance and partial (matched filter) covariance distributions.
  """
  n=1.2
  s = signal_levels[-1]
  N = correlation_trial_count_list[-1]
  K=20

  w = s + n * np.random.randn(N, K) # All the trial data

  full_cov = np.zeros((N*N, K))
  mf_cov = np.zeros((N, K))

  for i in range(K): # Iterate over experiments
    prod = w[:, i:i+1] * w[:, i:i+1].T
    assert prod.shape == (N, N)
    full_cov[:, i] = prod.reshape(N*N)

    mf_prod = w[:, i] * np.mean(w[:, i], axis=0)
    mf_cov[:, i] = mf_prod.reshape(N)

  plt.figure(figsize=figsize)

  bins = 50
  counts, edges = np.histogram(full_cov.reshape(-1), bins=50)
  counts = counts.astype(float) / np.max(counts)
  plt.plot((edges[:-1]+edges[1:])/2, counts, label='Full Covariance')

  counts, edges = np.histogram(mf_cov.reshape(-1), bins=50)
  counts = counts.astype(float) / np.max(counts)
  plt.plot((edges[:-1]+edges[1:])/2, counts, label='Partial Covariance')
  plt.legend();
  plt.xlim(-3, 30)
  plt.title('Full vs. Partial Covariance Distributions')
  plt.xlabel('Covariance')
  plt.ylabel('Normalized Histogram');

  print(f'Full mean: theory {s**2+n**2/N}, simulation {np.mean(full_cov.reshape(-1))}')
  print(f'Full var: theory {4*s**2*n**2 + (N+1)*n**4/(N**2)}, simulation {np.var(full_cov.reshape(-1))}')
  print(f'Full var2: theory {4*(s**2)*(n**2) + (N**2 - N + 2)*(n**4)/N**2}, simulation {np.var(full_cov.reshape(-1))}')
  print(f'Partial mean: theory {s**2+n**2/N}, simulation {np.mean(mf_cov.reshape(-1))}')
  print(f'Partial var: theory {(1+3/N)*s**2*n**2 + (N+1)*n**4/N**2}, simulation {np.var(mf_cov.reshape(-1))}')

  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)


######################## Single Point Power Metric Simulation ########################

# Single point power measures from the MMA notebook.

def spp_mean(s, n, N):
  return s*s + n*n/N

def spp_var(s, n, N):
  return 4*s*s*n*n/N+2*n**4/N**2

def spp_dprime(s, n, N):
  N = np.asarray(N, dtype=float)
  return 2*s/np.sqrt(2*s*s*n*n/N + 2*n**4/N**2)

def spp_threshold(s, n, d, numtrials):
  return ( numtrials )**( -1 ) * ( ( ( d )**( 2 ) * ( n )**( 2 ) * numtrials + \
d * ( ( 2 + ( d )**( 2 ) ) )**( 1/2 ) * ( n )**( 2 ) * numtrials ) )**( \
1/2 )


def plot_spp_stats(spp_mean_signal, spp_var_signal, spp_dprimes, 
                   figsize=(6.4, 4.8), 
                   plot_file='SinglePointPowerStats.png',
                   plot_dir: str = '.'):
  plt.figure(figsize=figsize)

  plt.subplot(3, 2, 1)
  plt.plot(signal_levels, np.asarray(spp_mean_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels, spp_mean(s, n, N), label='Theory')
  plt.title(f'Power {N} trials')
  plt.ylabel('Mean')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 2);
  plt.semilogx(correlation_trial_count_list, np.asarray(spp_mean_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spp_mean(s, n, N), label='Theory')
  plt.axhline(s*s, ls='--', label='Asymptote')
  plt.title(f'Power s={s:4.2f}');
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the Variances  #####################
  plt.subplot(3, 2, 3)
  plt.plot(signal_levels, np.asarray(spp_var_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels, 4*s*s*n*n/N+2*n**4/N**2, label='Theory')
  plt.ylabel('Variance')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 4);
  plt.semilogx(correlation_trial_count_list, np.asarray(spp_var_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, 4*s*s*n*n/N+2*n**4/N**2, label='Theory')
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the d's #####################
  plt.subplot(3, 2, 5) # Plot by signal level
  plt.plot(signal_levels, np.asarray(spp_dprimes)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list[-1]
  dprimes = s*s/np.sqrt(2*s*s*n*n/N + 2*n**4/N**2)
  plt.loglog(signal_levels, dprimes, label='Theory')
  # plt.plot(signal_levels, np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.axhline(spp_d_threshold, ls=':', label='d\' Threshold')
  plt.legend()
  plt.xlabel('Signal Level')
  plt.ylabel('d\'')

  plt.subplot(3, 2, 6)  # Plot by trial count
  plt.loglog(correlation_trial_count_list, np.asarray(spp_dprimes)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level  # Implicit definition used in the code above
  d = spp_d_threshold
  N = correlation_trial_count_list
  dprimes = s*s/np.sqrt(2*s*s*n*n/N + 2*n**4/N**2)
  plt.loglog(N, dprimes, label='Theory')
  # plt.axhline(np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.legend();
  plt.xlabel('Number of Trials');

  plt.axis('tight')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)


######################## Single Point Covariance Metric Simulation ########################

def spc_theory_mean(s, n, N):
  return s*s + n*n/N

def spc_theory_var(s, n, numtrials):
  numtrials = np.asarray(numtrials, dtype=float)
  return ( ( n )**( 4 ) * ( numtrials )**( -2 ) * ( 1 + numtrials ) + ( n )**( \
2 ) * ( 1 + 3 * ( numtrials )**( -1 ) ) * ( s )**( 2 ) )

def spc_theory_dprime(s, n, numtrials):
  return ( 2 )**( 1/2 ) * numtrials * ( s )**( 2 ) * ( ( 2 * ( n )**( 4 ) * ( \
1 + numtrials ) + ( n )**( 2 ) * numtrials * ( 3 + numtrials ) * ( s )**( \
2 ) ) )**( -1/2 )


def plot_spc_stats(spc_mean_signal, spc_var_signal, spc_dprimes, 
                   plot_file='SinglePointMatchedFilterStats.png',
                   figsize=(6.4, 4.8),
                   plot_dir: str = '.'):
  n = default_noise_level

  plt.figure(figsize=figsize))

  plt.subplot(3, 2, 1)
  plt.plot(signal_levels, np.asarray(spc_mean_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spc_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels,
          spc_theory_mean(signal_levels, default_noise_level, N),
          label='Theory')
  plt.title(f'Matched Filter {N} trials')
  plt.ylabel('Mean')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 2);
  plt.semilogx(correlation_trial_count_list, np.asarray(spc_mean_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spc_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spc_theory_mean(s, n, N), label='Theory')
  plt.axhline(s*s, ls='--', label='Asymptote')
  plt.title(f'Matched Filter s={s:4.2f}');
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the Variances  #####################
  plt.subplot(3, 2, 3)
  plt.plot(signal_levels, np.asarray(spc_var_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spc_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels, spc_theory_var(s, n, N), label='Theory')
  plt.ylabel('Variance')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 4);
  plt.semilogx(correlation_trial_count_list, np.asarray(spc_var_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spc_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spc_theory_var(s, n, N), label='Theory')
  plt.axhline(s*s*n*n, label='Asymptote')
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the d's #####################
  plt.subplot(3, 2, 5) # Plot by signal level
  plt.plot(signal_levels, np.asarray(spc_dprimes)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level  # Implicit definition used in the code above
  d = spc_d_threshold
  N = correlation_trial_count_list[-1]
  # dprimes = np.sqrt(2)*N*s**2/n/np.sqrt(N**2*s**2 + 3*N*s**2 + 2*N*n**2 + 2*n**2)
  dprimes = spc_theory_dprime(s, n, N)
  plt.plot(signal_levels, dprimes, label='Theory')
  # plt.plot(signal_levels, np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.axhline(spc_d_threshold, ls=':', label='d\' Threshold')
  plt.legend()
  plt.xlabel('Signal Level')
  plt.ylabel('d\'')

  plt.subplot(3, 2, 6)  # Plot by trial count
  plt.semilogx(correlation_trial_count_list, np.asarray(spc_dprimes)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level  # Implicit definition used in the code above
  d = spc_d_threshold
  N = correlation_trial_count_list
  # dprimes = np.sqrt(2)*N*s**2/n/np.sqrt(N**2*s**2 + 3*N*s**2 + 2*N*n**2 + 2*n**2)
  dprimes = spc_theory_dprime(s, n, N)
  plt.semilogx(N, dprimes, label='Theory')
  plt.axhline(np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.legend();
  plt.xlabel('Number of Trials');

  plt.axis('tight')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

######################## Single Point Full Covariance Metric Simulation ########################

def spf_theory_mean(s, n, N):
  return s*s + n*n/N

def spf_theory_var(s, n, N):
  N = np.asarray(N, dtype=float)  # Not used here except for the size
  # return ( ( n )**( 4 ) * ( numtrials )**( -2 ) * ( 1 + numtrials ) + 4 * ( n )**( 2 ) * ( s )**( 2 ) )
  # return 4*(s**2)*(n**2) + (numtrials**2 + 2*numtrials - 1)*(n**4)/numtrials**2
  # Result below due to Gemini... not sure why yet.
  # return 2*(s**2)*(n**2) + (N**2 + 2*N - 1)*(n**4)/N**2
  return n**4 + 2*(s**2)*(n**2) * np.ones(N.shape)

def spf_theory_dprime(s, n, N):
  N = np.asarray(N, dtype=float)
  # return numtrials * ( s )**( 2 ) * ( ( ( n )**( 4 ) * ( 1 + numtrials ) + 2 * ( n )**( 2 ) * ( numtrials )**( 2 ) * ( s )**( 2 ) ) )**( -1/2 )
  return (spf_theory_mean(s, n, N) - spf_theory_mean(0, n, N))/(
      np.sqrt((spf_theory_var(s, n, N) + spf_theory_var(0, n, N))/2))


def plot_spf_stats(spf_mean_signal, spf_var_signal, spf_dprimes, 
                   figsize=(6.4, 4.8), 
                   plot_dir: str = '.',
                   plot_file='SinglePointFullCovarianceStats.png'):
  n = default_noise_level

  plt.figure(figsize=figsize)

  plt.subplot(3, 2, 1)
  plt.plot(signal_levels, np.asarray(spf_mean_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spf_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels,
          spf_theory_mean(signal_levels, default_noise_level, N),
          label='Theory')
  plt.title(f'Full Covariance {N} trials')
  plt.ylabel('Mean')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 2);
  plt.semilogx(correlation_trial_count_list, np.asarray(spf_mean_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spf_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spf_theory_mean(s, n, N), label='Theory')
  plt.axhline(s*s, ls='--', label='Asymptote')
  plt.title(f'Full Covariance s={s:4.2f}');
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the Variances  #####################
  plt.subplot(3, 2, 3)
  plt.plot(signal_levels, np.asarray(spf_var_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spf_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels, spf_theory_var(s, n, N), label='Theory')
  plt.ylabel('Variance')
  plt.legend()

  plt.subplot(3, 2, 4);
  plt.semilogx(correlation_trial_count_list, np.asarray(spf_var_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spf_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spf_theory_var(s, n, N), label='Theory')
  # plt.axhline(s*s*n*n, label='Asymptote')
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the d's #####################
  plt.subplot(3, 2, 5) # Plot by signal level
  plt.plot(signal_levels, np.asarray(spf_dprimes)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level  # Implicit definition used in the code above
  d = spf_d_threshold
  N = correlation_trial_count_list[-1]
  dprimes = spf_theory_dprime(s, n, N)
  plt.plot(signal_levels, dprimes, label='Theory')
  # plt.plot(signal_levels, np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.axhline(spf_d_threshold, ls=':', label='d\' Threshold')
  plt.legend()
  plt.xlabel('Signal Level')
  plt.ylabel('d\'')

  plt.subplot(3, 2, 6)  # Plot by trial count
  plt.semilogx(correlation_trial_count_list, np.asarray(spf_dprimes)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level  # Implicit definition used in the code above
  d = spc_d_threshold
  N = correlation_trial_count_list
  dprimes = spf_theory_dprime(s, n, N)
  plt.semilogx(N, dprimes, label='Theory')
  # plt.axhline(np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.legend();
  plt.xlabel('Number of Trials');

  plt.axis('tight')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

##################### Single Point Jackknife (SPJ Stats #####################
# spj is single point correlation via jackknife


def spj_theory_mean(s, n, N):
  return s*s

def spj_theory_var(s, n, numtrials):
  numtrials = np.asarray(numtrials, dtype=float)
  return ( ( n )**( 4 ) * ( ( -1 + numtrials ) )**( -1 ) + ( n )**( 2 ) * ( 1 + \
( ( -1 + numtrials ) )**( -1 ) ) * ( s )**( 2 ) )

def spj_theory_dprime(s, n, numtrials):
  n = np.asarray(n, dtype=float)
  numtrials = np.asarray(numtrials, dtype=float)
  return ( 2 )**( 1/2 ) * ( n )**( -1 ) * ( ( -1 + numtrials ) )**( 1/2 ) * ( \
s )**( 2 ) * ( ( 2 * ( n )**( 2 ) + numtrials * ( s )**( 2 ) ) )**( \
-1/2 )


def plot_spj_stats(spj_mean_signal, spj_var_signal, spj_dprimes, 
                   figsize=(6.4, 4.8), 
                   plot_file='SinglePointJackknifeStats.png', 
                   plot_dir: str = '.'):
  n = default_noise_level

  plt.figure(figsize=figsize)
  plt.subplot(3, 2, 1)
  plt.plot(signal_levels, np.asarray(spj_mean_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spjk_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels,
          spj_theory_mean(signal_levels, default_noise_level, N),
          label='Theory')
  plt.title(f'Jackknife {N} trials')
  plt.ylabel('Mean')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 2);
  plt.semilogx(correlation_trial_count_list, np.asarray(spj_mean_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spjk_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spj_theory_mean(s, n, N)*np.ones(len(N)), label='Theory')
  plt.axhline(s*s, ls=':', label='Asymptote')
  plt.title(f'Jackknife s={s:4.2f}');
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the Variances  #####################
  plt.subplot(3, 2, 3)
  plt.plot(signal_levels, np.asarray(spj_var_signal)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  d = spjk_d_threshold
  N = correlation_trial_count_list[-1]
  plt.plot(signal_levels, spj_theory_var(s, n, N), label='Theory')
  plt.ylabel('Variance')
  plt.gca().set(xticklabels=[])
  plt.legend()

  plt.subplot(3, 2, 4);
  plt.semilogx(correlation_trial_count_list, np.asarray(spj_var_signal)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  d = spjk_d_threshold
  N = correlation_trial_count_list
  plt.semilogx(N, spj_theory_var(s, n, N), label='Theory')
  plt.axhline(s*s*n*n, label='Asymptote', ls=':')
  plt.gca().set(xticklabels=[])
  plt.legend()

  ##################### Now plot the d's #####################
  plt.subplot(3, 2, 5) # Plot by signal level
  plt.plot(signal_levels, np.asarray(spj_dprimes)[:, -1], 'x', label='Simulation')
  s = np.asarray(signal_levels)
  n = default_noise_level  # Implicit definition used in the code above
  d = spjk_d_threshold
  N = correlation_trial_count_list[-1]
  # dprimes = np.sqrt(2)*N*s**2/n/np.sqrt(N**2*s**2 + 3*N*s**2 + 2*N*n**2 + 2*n**2)
  dprimes = spj_theory_dprime(s, n, N)
  plt.plot(signal_levels, dprimes, label='Theory')
  # plt.plot(signal_levels, np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.axhline(spjk_d_threshold, ls=':', label='d\' Threshold')
  plt.legend()
  plt.xlabel('Signal Level')
  plt.ylabel('d\'')

  plt.subplot(3, 2, 6)  # Plot by trial count
  plt.semilogx(correlation_trial_count_list, np.asarray(spj_dprimes)[-1, :], 'x', label='Simulation')
  s = np.asarray(signal_levels)[-1]
  n = default_noise_level  # Implicit definition used in the code above
  d = spjk_d_threshold
  N = correlation_trial_count_list
  # dprimes = np.sqrt(2)*N*s**2/n/np.sqrt(N**2*s**2 + 3*N*s**2 + 2*N*n**2 + 2*n**2)
  dprimes = spj_theory_dprime(s, n, N)
  plt.semilogx(N, dprimes, label='Theory')
  plt.axhline(np.sqrt(2)*s/n, ls=':', label='Asymptote')
  plt.legend();
  plt.xlabel('Number of Trials');

  plt.axis('tight')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

##################### Colored Noise - Fouier Approach #####################
def make_basis(N) -> NDArray:
  """Make a set of cosine basis functions"""
  assert np.floor(np.log2(N)) == np.log2(N)

  basis = np.zeros((N, N))
  for i in range(N):
    basis[:, i] = np.cos(2*np.pi*i*np.arange(N)/N)
  return basis

def make_signal(spectrum: NDArray) -> NDArray:
  """Given a Fourier spectrum, invert the FFT to get a real time-domain waveform."""
  N = len(spectrum)
  return np.real(np.fft.ifft(spectrum))*N

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

def make_noise(N: int, noise_level: float) -> NDArray:
  """Create a noise spectrum and invert it to get a real noisy waveform."""
  spectrum = noise_level * np.exp(1j * np.random.rand(N//2+1)*2*np.pi)
  spectrum[0] = noise_level
  spectrum[N//2] = noise_level
  spectrum = make_hermitian(spectrum)
  return spectrum, N*np.real(np.fft.ifft(spectrum))


def make_experiment(signal_waveform: NDArray,
                    num_trials: int,
                    noise_level: float = 1) -> NDArray:
  """Create a set of trials that make up a full experiment."""
  N = len(signal_waveform)
  results = np.zeros((N, num_trials))
  for i in range(num_trials):
    noise_spectrum, noise_waveform = make_noise(N, noise_level)
    results[:, i] = signal_waveform + noise_waveform
  return results, noise_spectrum

def EstimateCorrelation(true_signal, results: NDArray)-> float:
  assert len(true_signal) == results.shape[0], f'Len(signal) is {len(true_signal)}, results.shape is {results.shape}'

  true_signal = np.reshape(true_signal, (-1, 1))

  correlation = np.sum(true_signal*results, axis=0)
  return np.mean(correlation), np.var(correlation)

def colored_theory_mean(signal_spectrogram, noise_spectrogram):
  assert signal_spectrogram.shape == noise_spectrogram.shape
  N = signal_spectrogram.shape[0]
  return N*np.sum(signal_spectrogram*np.conjugate(signal_spectrogram))

def colored_theory_var(signal_spectrogram, noise_spectrogram):
  assert signal_spectrogram.shape == noise_spectrogram.shape
  N = signal_spectrogram.shape[0]
  return N*N*np.sum(signal_spectrogram*np.conjugate(signal_spectrogram)*
                      noise_spectrogram*np.conjugate(noise_spectrogram))


def colored_noise_simulation(figsize=(6.4, 4.8), 
                             plot_file: str = 'ColoredNoiseResult.png',
                             plot_dir: str = '.'):
  N = 128
  signal_spectrum = np.zeros(N)
  signal_spectrum[4] = 1
  signal_spectrum[-4] = 1

  np.random.seed(FLAGS.seed)
  noises = np.asarray([0.25, 0.5, 1, 2, 4])/4
  signal_level = 3
  sim_means = []
  sim_vars = []
  theory_means = []
  theory_vars = []
  for n in noises:
    test_spectrum = signal_level * signal_spectrum
    test_signal = make_signal(test_spectrum)
    results, noise_spectrum = make_experiment(test_signal, 1000, noise_level=n)
    mean, var = EstimateCorrelation(test_signal, results)
    sim_means.append(mean)
    sim_vars.append(var)
    theory_means.append(colored_theory_mean(test_spectrum, noise_spectrum))
    theory_vars.append(colored_theory_var(test_spectrum, noise_spectrum))
  sim_means = np.asarray(sim_means)
  sim_vars = np.asarray(sim_vars)
  theory_means = np.asarray(theory_means)
  theory_vars = np.asarray(theory_vars)

  plt.figure(figsize=figsize)
  plt.loglog(noises, theory_means, label='Theoretical Mean')
  plt.loglog(noises, sim_means, 'x', label='Simulated Mean')
  plt.loglog(noises, theory_vars, label='Theoretical Variance')
  plt.loglog(noises, sim_vars, 'o', label='Simulated Variance')
  plt.xlabel('Noise Level');
  plt.legend()
  plt.title('Colored Noise Simulation')
  plt.ylabel('Variance of Correlation Measure')
  np.mean(sim_means / theory_means), np.mean(sim_vars / theory_vars)

  plt.savefig(os.path.join(plot_dir, plot_file))

##################### Waveform Stack Example    #####################
mouse_sample_rate = 16000

def create_synthetic_stack(noise_level=1, num_times=1952, num_trials=1026,
                           bw=200, order=4, cf=1000, signal_levels=(0, 1),
                           sample_rate=mouse_sample_rate):
    """Create a synthetic stack of ABR recordings so we can investigate d'
    behaviour for really large number of trials.  This stack has two different
    (sound pressure) levels.

    Args:
      noise_level: Amplitude of the white noise signal.
      num_times: The length of the ABR response in samples
      num_trials: How many trials to create
      bw: The bandwidth of the envelope (in Hz)
      order: the Gammatone order
      cf: The center frequency of the carrier (in Hz)
      signal_levels: A list of signal levels to generate
      sample_rate: What sample rate to generate the signal at

    Returns:
      a 3d tensor with shape num_levels x num_times x num_trials
    """
    def gammatone_func(t, cf=cf, bw=bw, order=order):
        envelope = t ** (order - 1) * np.exp(-2 * np.pi * bw * t)
        if cf:
          return envelope * np.sin(2 * np.pi * cf * t)
        else:
          return envelope

    np.random.seed(FLAGS.seed)
    t = np.arange(num_times) / mouse_sample_rate
    peak_time = 3/(2*np.pi*bw)
    peak_env = gammatone_func(peak_time, cf=0)
    gammatone = gammatone_func(t)/peak_env
    # The shape of the stacks array is levels x time x trials
    stack = noise_level * np.random.normal(size=(len(signal_levels), num_times, num_trials))
    signals = np.expand_dims(signal_levels, (1, 2)) * np.expand_dims(gammatone, (0, 2))
    stack += signals
    return stack

def plot_synthetic_stack_example(figsize=(6.4, 4.8), 
                                 plot_file='WaveformStack.png',
                                 plot_dir: str = '.'):
  plt.figure(figsize=figsize)
  stack = create_synthetic_stack(0.05, 1952, 10);
  num_points = 150
  num_waveforms = stack.shape[-1]
  for i in reversed(range(num_waveforms)):
    times = np.arange(num_points) + 5*i
    plt.plot(times, stack[1, :num_points, i] + .4*i,
             alpha=(num_waveforms-i)/num_waveforms)
  plt.gca().annotate('', xytext=(num_points + 10, 0),
              xy=(num_points + 5*num_waveforms, .4*(num_waveforms-1)),
              arrowprops=dict(arrowstyle="->"))
  plt.gca().annotate('', xytext=(0, -1.1),
                    xy = (num_points, -1.1),
                    arrowprops=dict(arrowstyle="->"))

  plt.text(180, 1.0, 'Number of \nTrials (N)')
  plt.text(75, -1.4, 'Time')
  plt.axis('off');
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

##################### Comparing d's    #####################

def compare_dprimes(figsize=(6.4, 4.8), 
                    plot_file: str = 'DPrimeComparison.png',
                    plot_dir: str = '.'):
  N = (2**np.arange(4, 8, 0.25)).astype(float)
  s = 3.1
  n = 1
  plt.figure(figsize=figsize)
  plt.loglog(N, spj_theory_dprime(s, n, N)*np.sqrt(N), label='Multilook Jackknife')
  plt.loglog(N, spp_dprime(s, n, N), label='Power of Average')
  plt.loglog(N, spj_theory_dprime(s, n, N), label='Jackknife (single trial)')
  plt.loglog(N, spc_theory_dprime(s, n, N), label='Matched Filter (single trial)')
  plt.loglog(N, spf_theory_dprime(s, n, N), label='Full Correlation (single trial)')
  plt.xlabel('Number of Trials (N)')
  plt.ylabel('d\'')
  plt.legend()
  plt.ylim(2, 50)
  plt.title('Comparison of Detection Methods')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

##################### Multilook Histograms #####################

def XXmultilook_plot(plot_dir: str = '.'):
  # Make a plot show how muliple looks are used.
  def gaussian_scatter(center=(0, 0), r=0.3, count=100) -> NDArray:
    points = np.random.randn(len(center), count)*r
    return points + np.expand_dims(np.asarray(center), axis=1)

  count = 4000
  points0 = gaussian_scatter(center=(1, 0), r=0.3, count=count)
  points1 = gaussian_scatter(center=(0, 1), r=0.3, count=count)
  points2 = gaussian_scatter(center=(1, 1), r=0.3, count=count)
  alpha = 0.05

  plt.figure(figsize=(10, 10))

  plt.subplot(2, 2, 1)
  counts, centers = np.histogram(points2[0,:], bins=30)
  plt.plot(counts, (centers[1:] + centers[:-1])/2);
  plt.xlabel('Frequency'), plt.ylabel('Correlation'), plt.title('Look 1 Distribution')
  plt.axhline(1.0, ls='--', color='r')
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['right'].set_visible(False)


  plt.subplot(2, 2, 3)
  plt.plot(points2[0, :], points2[1, :], '.', alpha=0.1)
  plt.title('Correlations for Both Looks')
  plt.xlabel('Look 2 Correlation'), plt.ylabel('Look 1 Correlation')
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['right'].set_visible(False)

  plt.subplot(2, 2, 2)
  distance = np.sqrt(points2[0, :]**2 + points2[1, :]**2)
  counts, centers = np.histogram(distance, bins=30)
  plt.plot((centers[1:] + centers[:-1])/2, counts)
  plt.axvline(np.sqrt(2), ls='--', color='r')
  plt.ylabel('Frequency'), plt.xlabel('Correlation'), plt.title('2 Look Distribution')
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['right'].set_visible(False)
  plt.annotate('Note larger mean', (1.414, 100), (1.85, 200),
              arrowprops=dict(arrowstyle='->', color='red'))

  plt.subplot(2, 2, 4)
  counts, centers = np.histogram(points2[1,:], bins=30)
  plt.plot((centers[1:] + centers[:-1])/2, counts)
  plt.axvline(1.0, ls='--', color='r');
  plt.ylabel('Frequency'), plt.xlabel('Correlation'), plt.title('Look 2 Distribution');
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['right'].set_visible(False)

  plt.savefig(os.path.join(plot_dir, 'MultilookDPrimeComparison.png'), dpi=300)


# Make a plot show how muliple looks are used.
def gaussian_scatter(center=(0, 0), r=0.3, count=100) -> NDArray:
  points = np.random.randn(len(center), count)*r
  return points + np.expand_dims(np.asarray(center), axis=1)


def multilook_plot(plot_dir: str = '.', figsize=(6.4, 4.8),
                   plot_file='MultilookDPrimeComparison.png'):
  count = 4000

  noise_points = gaussian_scatter(center=(0, 0), r=0.3, count=count)
  signal_points = gaussian_scatter(center=(1, 1), r=0.3, count=count)

  alpha = 0.1

  def show_dist(second_mean, label: str = '', y=250, color='k', ):
    plt.axvline(0, ls='--', color=color)
    plt.axvline(second_mean, ls='--', color=color)
    plt.annotate("", (0, y), (second_mean, y),
                      arrowprops=dict(arrowstyle="<->", color='black'))
    plt.text(0.4*second_mean, y+10, label)

  plt.figure(figsize=figsize)
  plt.subplot(2, 2, 1)
  plt.hist(noise_points[0, :], bins=30, label='Noise')
  plt.hist(signal_points[0, :], bins=30, label='Signal')
  plt.xlim(-1, 2)
  show_dist(1, '1.0')
  plt.legend()
  plt.title('Histogram of Look 1')

  plt.subplot(2, 2, 2)
  projector = np.ones((2, count))/np.sqrt(2)
  noise_projection = np.sum(noise_points * projector, axis=0)
  signal_projection = np.sum(signal_points * projector, axis=0)
  plt.hist(noise_projection, bins=30, label='Noise')
  plt.hist(signal_projection, bins=30, label='Signal')
  plt.xlim(-1, 3)
  show_dist(np.sqrt(2), '1.414')
  plt.legend()
  plt.title('Histogram of Projection')

  plt.subplot(2, 2, 3)
  plt.plot(noise_points[0, :], noise_points[1, :], '.', alpha=alpha)
  plt.plot(signal_points[0, :], signal_points[1, :], '.', alpha=alpha);
  plt.plot([-1, 2], [-1, 2], '--')
  plt.xlim(-1, 2)
  plt.ylim(-1, 2)
  plt.title('Scatter of Measurements')

  plt.subplot(2, 2, 4)
  plt.hist(noise_points[1, :], bins=30, label='Noise')
  plt.hist(signal_points[1, :], bins=30, label='Signal')
  plt.xlim(-1, 2)
  show_dist(1, '1.0')
  plt.legend();
  plt.title('Histogram of Look 2');

  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

def threshold_theory_ratio(figsize=(6.4, 4.8), 
                           plot_file='AverageVsCovarianceThresholdRatio.png',
                           plot_dir: str = '.'):
  def ratio(d):
    return 2 * ( ( d + ( ( 2 + ( d )**( 2 ) ) )**( 1/2 ) ) )**( 1/2 ) * ( ( d + ( \
           ( 16 + ( d )**( 2 ) ) )**( 1/2 ) ) )**( -1/2 )

  plt.figure(figsize=figsize)
  d = 10**(np.arange(-2, 2, .01))
  plt.semilogx(d, ratio(d))
  plt.xlabel('d\'')
  plt.ylabel('Power of average threshold / covariance threshold')
  plt.title('Comparison of sound threshold for power of average vs. covariance')
  plt.savefig(os.path.join(plot_dir, plot_file), dpi=300)

##################### Main Program  #####################


def main(_argv=None):
  # compare_full_partial_correlation()
  print('ABR Simulations. Cache dir is', FLAGS.cache_dir)

  (spp_dprimes, spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
   spc_dprimes, spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
   spf_dprimes, spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal, 
   signal_levels) = get_simulation_data(num_experiments=FLAGS.num_experiments,
                                        cache_dir=FLAGS.cache_dir,
                                        jackknife=False)

  plot_spp_stats(spp_mean_signal, spp_var_signal, spp_dprimes, plot_dir=FLAGS.cache_dir)
  plot_spc_stats(spc_mean_signal, spc_var_signal, spc_dprimes, plot_dir=FLAGS.cache_dir)
  plot_spf_stats(spf_mean_signal, spf_var_signal, spf_dprimes, plot_dir=FLAGS.cache_dir)

  (_, _, _, _, _,
   spj_dprimes, spj_mean_noise, spj_mean_signal, spj_var_noise, spj_var_signal,
   _, _, _, _, _,
   signal_levels) = get_simulation_data(
            num_experiments=FLAGS.num_jackknife_experiments,
            cache_dir=FLAGS.cache_dir, 
            jackknife=True)
  plot_spj_stats(spj_mean_signal, spj_var_signal, spj_dprimes, plot_dir=FLAGS.cache_dir)

  compare_dprimes(plot_dir=FLAGS.cache_dir)
  multilook_plot(plot_dir=FLAGS.cache_dir)
  threshold_theory_ratio(plot_dir=FLAGS.cache_dir)
  plot_synthetic_stack_example(plot_dir=FLAGS.cache_dir)
  colored_noise_simulation(plot_dir=FLAGS.cache_dir)

if __name__ == '__main__':
   app.run(main)