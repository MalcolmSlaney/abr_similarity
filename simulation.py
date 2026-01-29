import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Tuple

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
from sklearn.linear_model import Ridge, LinearRegression


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

correlation_trial_count_list


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

matched_filter_correlate(np.random.randn(3, 4))


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

full_correlate(np.random.randn(3, 4))

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
    # num_experiments = 1000
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

    return (spc_dprimes, spf_dprimes, spp_dprimes,
            spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
            spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
            spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal)


# Just do the highest signal level, and the highest trial count,
# to make sure the code works.

# spc is single point correlation
# spf is single point full correlation
# spp is single point power metric
# spjk is single point correlation via jackknife

(spc_dprimes, spf_dprimes, spp_dprimes,
 spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
 spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
 spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal
 ) = simulate_point_process(n=default_noise_level, num_experiments=20,
                            signal_levels=signal_levels[-1:],
                            correlation_trial_count_list=correlation_trial_count_list[-1:])



# Now do the full simulation, across all signal levels and trial counts

# spc is single point correlation
# spf is single point full correlation
# spp is single point power metric
# spjk is single point correlation via jackknife

(spc_dprimes, spf_dprimes, spp_dprimes,
 spc_mean_noise, spc_mean_signal, spc_var_noise, spc_var_signal,
 spp_mean_noise, spp_mean_signal, spp_var_noise, spp_var_signal,
 spf_mean_noise, spf_mean_signal, spf_var_noise, spf_var_signal
 ) = simulate_point_process(n=default_noise_level, num_experiments=20)


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

plt.figure(figsize=(4, 3))

bins = 50
counts, edges = np.histogram(full_cov.reshape(-1), bins=50)
counts = counts.astype(float) / np.max(counts)
plt.plot((edges[:-1]+edges[1:])/2, counts, label='Full Correlation')

counts, edges = np.histogram(mf_cov.reshape(-1), bins=50)
counts = counts.astype(float) / np.max(counts)
plt.plot((edges[:-1]+edges[1:])/2, counts, label='Partial Correlation')
plt.legend();
plt.xlim(-3, 30)
plt.title('Full vs. Partial Correlation Distributions')
plt.xlabel('Correlation')
plt.ylabel('Normalized Histogram');

print(f'Full mean: theory {s**2+n**2/N}, simulation {np.mean(full_cov.reshape(-1))}')
print(f'Full var: theory {4*s**2*n**2 + (N+1)*n**4/(N**2)}, simulation {np.var(full_cov.reshape(-1))}')
print(f'Full var2: theory {4*(s**2)*(n**2) + (N**2 - N + 2)*(n**4)/N**2}, simulation {np.var(full_cov.reshape(-1))}')
print(f'Partial mean: theory {s**2+n**2/N}, simulation {np.mean(mf_cov.reshape(-1))}')
print(f'Partial var: theory {(1+3/N)*s**2*n**2 + (N+1)*n**4/N**2}, simulation {np.var(mf_cov.reshape(-1))}')