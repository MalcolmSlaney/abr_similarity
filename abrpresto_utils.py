import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import polynomial as poly


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


def XXget_level_for_dprime(levels, dprimes, target_dprime):
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

  def XXrms_error(self, levels: ArrayLike, dprimes: ArrayLike) -> float:
    """Calculate mean squared error between the fitted sigmoid and the actual data."""
    if np.isnan(self.L_fit):
      return float('nan')
    dprimes_fit = self._sigmoid_func(levels, self.L_fit, self.x0_fit, self.k_fit, self.b_fit)
    mse = np.mean((dprimes - dprimes_fit) ** 2)
    return mse / len(levels) 

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

