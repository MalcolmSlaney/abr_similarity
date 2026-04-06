# This file contains the 
# 1) curve-fitting algorithms needed to smooth the correlation data, and 
# 2) EEG Waveform preprocessing code (bandpass and time window)

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter
from numpy import polynomial as poly

################# Curve Fitting Code ############################


class FitCurve(object):
  """Base class for fitting curves to d' vs. level data."""
  def __init__(self, levels: ArrayLike, dprimes: ArrayLike, plot=False):
    raise NotImplementedError("Subclasses should implement this method.")

  def compute(self, levels: ArrayLike) -> ArrayLike:
    raise NotImplementedError("Subclasses should implement this method.")

  def inverse_compute(self, target_dprime: float) -> float:
    raise NotImplementedError("Subclasses should implement this method.")

  def rms_error(self, levels: ArrayLike, dprimes: ArrayLike) -> float:
    """Calculate mean squared error between the fitted curve and the actual data."""
    computed_dprimes = self.compute(levels)
    mse = np.mean((dprimes - computed_dprimes) ** 2)
    return mse / len(levels)

class FitQuadraticMonomialCurve(FitCurve):
  """Fit a quadratic monomial curve (d' = a * level^2) to the data
  """
  def __init__(self, levels: ArrayLike, dprimes: ArrayLike, order: int = 2, plot=False):
    self.coefficients = poly.polynomial.polyfit(levels, dprimes, deg=[2,])
    if plot:
      plt.figure(figsize=(10, 6))
      plt.plot(levels, dprimes, 'o', label='Original Data')
      plt.plot(levels, self.compute(levels), '-', label='Quadratic Fit')
      plt.legend()
      plt.title('Quadratic Fit to d\' vs. Level')
      plt.xlabel('Level (dB)')
      plt.ylabel('d\'')
      plt.grid(True)

  def compute(self, levels: ArrayLike) -> ArrayLike:
    result = 0
    for i, coeff in enumerate(self.coefficients):
      result += coeff * levels**i
    return result

  def inverse_compute(self, target_dprime: float) -> float:
    """Given a d-prime value, return the corresponding level.

    Solves a*x^2 + b*x + (c - target_dprime) = 0 for x.
    """
    # Coefficients are stored as [c, b, a]
    c_val, b_val, a_val = self.coefficients

    # Form the quadratic equation a_val*x^2 + b_val*x + (c_val - target_dprime) = 0
    c_prime = c_val - target_dprime

    discriminant = b_val**2 - 4 * a_val * c_prime

    if a_val == 0:
        # Linear case: bx + c_prime = 0 => x = -c_prime / b
        if b_val == 0:
            return float('nan') # No solution or infinite solutions
        return -c_prime / b_val
    
    if discriminant < 0:
      return float('nan') # No real roots
    elif discriminant == 0:
      return -b_val / (2 * a_val) # One real root
    else:
      # Two real roots. Choose the one that corresponds to increasing d' with level.
      # For ABR, d' generally increases with level, so we typically want the higher level.
      root1 = (-b_val + np.sqrt(discriminant)) / (2 * a_val)
      root2 = (-b_val - np.sqrt(discriminant)) / (2 * a_val)

      # Assuming 'a_val' is generally positive for ABR (parabola opens upwards)
      # and we're interested in the increasing part of the curve.
      if a_val > 0:
          return max(root1, root2)
      else: # Parabola opens downwards
          return min(root1, root2) # Take the smaller root on the increasing side


class FitPowerCurve(FitCurve):
  """
  Fit a power curve above a breakpoint, of the form
    d' = a * (level - breakpoint)^2 for level > breakpoint, else 0
  to the data
  """
  def piecewise_func(self, level, breakpoint, a):
    # If level < breakpoint, return 0. Otherwise, return a * level^2
    # Changed np.max to np.maximum for element-wise comparison
    level_after_breakpoint = np.maximum(0.0, level - breakpoint)
    return a * level_after_breakpoint**2

  def __init__(self, levels: ArrayLike, dprimes: ArrayLike, order: int = 2, plot=False):
    # 2. Fit the function to the data
    # We provide initial guesses for [breakpoint, a] using the p0 argument.
    initial_guess = [5.0, 1.0]

    # popt contains the optimal parameters; pcov is the covariance matrix
    try:
      popt, pcov = curve_fit(self.piecewise_func, levels, dprimes, p0=initial_guess)
    except RuntimeError as e:
      print(f"Error fitting curve: {e}", levels, dprimes)
      popt = 0, 0
    self.fitted_breakpoint, self.fitted_a = popt

    if plot:
      plt.plot(levels, dprimes, 'o', label='Original Data')
      plt.plot(levels, self.piecewise_func(levels, self.fitted_breakpoint,
                                           self.fitted_a), '-', label='Fitted Curve')
      plt.legend()
      plt.title('Power Curve Fit to d\' vs. Level')
      plt.xlabel('Level (dB)')
      plt.ylabel('d\'')

  def compute(self, levels:ArrayLike) -> ArrayLike:
    return self.piecewise_func(levels, self.fitted_breakpoint, self.fitted_a)

  def inverse_compute(self, target_dprime: float) -> float:
    """Given a d-prime value, return the corresponding level."""
    if target_dprime <= 0:
      return self.fitted_breakpoint

    if self.fitted_a <= 0:
      return float('nan') # 'a' should be positive for increasing d' with level

    arg_sqrt = target_dprime / self.fitted_a
    if arg_sqrt < 0:
      return float('nan')

    level_diff = np.sqrt(arg_sqrt)
    estimated_level = self.fitted_breakpoint + level_diff
    return estimated_level


class FitSigmoidCurve(FitCurve):
  """Fit a sigmoid curve to the data, of the form
    d' = L / (1 + exp(-k*(level - x0))) + b
  where L is the maximum value of the sigmoid, x0 is the midpoint, k is the steepness, and b is the baseline.
  """
  @staticmethod
  def _sigmoid_func(x, L, x0, k, b):
    """Sigmoid function."""
    return L / (1 + np.exp(-k * (x - x0))) + b

  def __init__(self, levels: ArrayLike, dprimes: ArrayLike, plot=False):
    # Initial guess for the parameters (L, x0, k, b)
    # L: maximum value of the sigmoid (approximate max of dprimes)
    # x0: midpoint of the sigmoid (approximate middle of levels)
    # k: steepness of the sigmoid
    # b: baseline offset
    # Ensure k is positive for a standard sigmoid shape
    initial_guess = [
        np.max(dprimes) - np.min(dprimes), # L
        np.mean(levels),                   # x0
        0.1,                               # k
        np.min(dprimes)                    # b
    ]

    try:
        # Fit the sigmoid function to the data
        popt, pcov = curve_fit(self._sigmoid_func, levels, dprimes, p0=initial_guess, maxfev=5000)
        self.L_fit, self.x0_fit, self.k_fit, self.b_fit = popt

        if plot:
          levels_smooth = np.linspace(min(levels), max(levels), 100)
          dprimes_fit = self._sigmoid_func(levels_smooth, self.L_fit, self.x0_fit, self.k_fit, self.b_fit)

          plt.figure(figsize=(10, 6))
          plt.plot(levels, dprimes, 'o', label='Original Data')
          plt.plot(levels_smooth, dprimes_fit, 'r-', label='Fitted Sigmoid Curve')
          plt.title("Sigmoid Fit to d' vs. Level")
          plt.xlabel('Level (dB)')
          plt.ylabel('d\'')
          plt.legend()
          plt.grid(True)

    except RuntimeError as e:
        print(f"Error fitting sigmoid: {e}")
        print("Could not find optimal parameters for the sigmoid function. The data might not be well-suited for a sigmoid fit or initial guess might need adjustment.")
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(levels, dprimes, 'o', label='Original Data (Fit Failed)')
            plt.title("Sigmoid Fit Failed to d' vs. Level")
            plt.xlabel('Level (dB)')
            plt.ylabel('d\'')
            plt.legend()
            plt.grid(True)
        self.L_fit, self.x0_fit, self.k_fit, self.b_fit = float('nan'), float('nan'), float('nan'), float('nan')

  def compute(self, levels: ArrayLike) -> ArrayLike:
    """Compute d-prime values for given levels using the fitted sigmoid."""
    return self._sigmoid_func(levels, self.L_fit, self.x0_fit, self.k_fit, self.b_fit)

  def inverse_compute(self, target_dprime: float) -> float:
    """Given a d-prime value, return the corresponding level."""
    if np.isnan(self.k_fit) or self.k_fit == 0:
      return float('nan')

    # x = x0 - log(L / (y - b) / k - 1)
    term = self.L_fit / (target_dprime - self.b_fit) - 1

    if term <= 0:
      return float('nan') # Logarithm of non-positive number

    estimated_level = self.x0_fit - np.log(term) / self.k_fit
    return estimated_level


################# ABRPresto Preprocessing Code ############################

def dataframe_fs(df: pd.DataFrame) -> float:
  """
  Calculate the sampling frequency of a DataFrame based on its column names, 
  which represent time points. Assumes that the columns are sorted in ascending 
  order.
  """
  # Extract the column names as an array
  time_array = df.columns.to_numpy()

  # Calculate the differences between consecutive time steps
  time_steps = np.diff(time_array)

  # Use the mean of the steps to account for any tiny floating-point inaccuracies
  dt = np.mean(time_steps)

  # Calculate sampling frequency (1 divided by the time step)
  return 1.0 / dt


def abrpresto_bandpass(
    data: pd.DataFrame,
    fs: float = 0,
    low_freq: float = 300.0,
    high_freq: float = 3000.0,
    order: int = 2,
    time_start: float = 0.0005, # seconds, equivalent to 0.5 ms
    time_end: float = 0.006, # seconds, equivalent to 6 ms
) -> pd.DataFrame:
    """
    Applies a bandpass filter and time-slices the data. The time-domain window,
    in particular, is important because there seems to be a glitch at the end of
    the recordings, and this results in a large false correlation.

    Args:
        data: DataFrame containing time series data. The columns are time points.
        fs: Sampling frequency in Hz.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        order: Order of the Butterworth filter.
        time_start: Start time for slicing in seconds.
        time_end: End time for slicing in seconds.

    Returns:
        A new DataFrame with filtered and time-sliced data.
    """
    if not fs:
      fs = dataframe_fs(data)
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')

    # Ensure filtered_data remains a DataFrame by explicitly returning a Series 
    # with original index
    filtered_data = data.apply(lambda x: pd.Series(lfilter(b, a, x.values), 
                                                   index=x.index), axis=1)

    # Slice the filtered data based on time_start and time_end
    # Find the indices corresponding to time_start and time_end
    time_columns = filtered_data.columns.to_numpy()
    start_idx = np.searchsorted(time_columns, time_start)
    end_idx = np.searchsorted(time_columns, time_end)

    # Ensure the indices are within bounds
    if start_idx == len(time_columns):
        start_idx = len(time_columns) - 1
    if end_idx == len(time_columns):
        end_idx = len(time_columns) - 1
    if start_idx > end_idx:
        start_idx = end_idx # In case of inverted or out-of-range times, adjust to avoid empty slice.

    # Slice the DataFrame
    sliced_filtered_data = filtered_data.iloc[:, start_idx:end_idx]

    return sliced_filtered_data

