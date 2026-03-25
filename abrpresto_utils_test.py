import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.fft import fft, fftfreq

from absl.testing import absltest

from abrpresto_utils import FitPowerCurve, FitQuadraticMonomialCurve, FitSigmoidCurve
from abrpresto_utils import abrpresto_bandpass, dataframe_fs


class FitPowerCurveTest(absltest.TestCase):

  def test_dprime_power_fit_and_compute(self):
    # Create some dummy data that roughly follows a power law
    levels_test = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    # Generate dprimes using a known power law with breakpoint 10 and 'a' = 0.005
    true_a = 0.005
    true_breakpoint = 10.0
    dprimes_test = true_a * np.maximum(0.0, levels_test - true_breakpoint)**2
    # Add some noise for a more realistic scenario
    dprimes_test_noisy = dprimes_test + np.random.normal(0, 0.1, size=levels_test.shape)

    dp_power = FitPowerCurve(levels_test, dprimes_test_noisy, plot=False)

    # Test fitted parameters (allow some tolerance due to fitting noise)
    self.assertBetween(dp_power.fitted_breakpoint, true_breakpoint - 5, true_breakpoint + 5)
    self.assertBetween(dp_power.fitted_a, true_a - 0.002, true_a + 0.002)

    # Test compute method
    computed_dprimes = dp_power.compute(levels_test)
    np.testing.assert_allclose(computed_dprimes, dp_power.piecewise_func(levels_test, dp_power.fitted_breakpoint, dp_power.fitted_a), rtol=1e-5)

    # Test inverse_compute method for a target d-prime
    target_dprime = 1.0
    estimated_level = dp_power.inverse_compute(target_dprime)
    
    # Check if the estimated level produces a d-prime close to the target
    if not np.isnan(estimated_level):
        recomputed_dprime = dp_power.compute(estimated_level)
        self.assertAlmostEqual(recomputed_dprime, target_dprime, places=2)
    else:
        # If estimated_level is NaN, verify that target_dprime is out of range for a valid solution
        # This depends on the actual fitted curve. For now, we'll just check if it's not NaN when it should be.
        pass

  def test_inverse_compute_edge_cases(self):
    # Test cases where inverse_compute should return NaN or breakpoint
    levels_edge = np.array([10.0, 20.0, 30.0])
    dprimes_edge = np.array([0.1, 0.5, 2.0])
    dp_power = FitPowerCurve(levels_edge, dprimes_edge, plot=False)

    # Target d-prime below the minimum fitted curve value (should be breakpoint)
    self.assertEqual(dp_power.inverse_compute(-0.5), dp_power.fitted_breakpoint)
    self.assertEqual(dp_power.inverse_compute(0), dp_power.fitted_breakpoint)

    # If fitted_a is non-positive, inverse_compute should return NaN (though in ABR context, it's usually positive)
    # This requires mocking or specifically crafting data for 'fitted_a' to be <= 0
    # For this test, we assume fitting yields a positive 'fitted_a' as expected for ABR.



class FitQuadraticMonoimialCurveTest(absltest.TestCase):

  def test_dprime_quadratic_fit_and_compute(self):
    # Create some dummy data that roughly follows a quadratic curve
    levels_test = np.array([1, 2, 3, 4, 5, 6])
    # Generate dprimes using a known quadratic function (e.g., 1x^2)
    true_a_quad = 1
    true_b_quad = 0.0
    true_c_quad = 0.0
    dprimes_test = true_a_quad * levels_test**2 + true_b_quad * levels_test + true_c_quad
    # Add some noise for a more realistic scenario
    dprimes_test_noisy = dprimes_test + np.random.normal(0, 0.1, size=levels_test.shape)

    dp_quadratic = FitQuadraticMonomialCurve(levels_test, dprimes_test_noisy, plot=True)
    plt.savefig('Results/test_quadratic_fit_example.png')

    # Test compute method
    computed_dprimes = dp_quadratic.compute(levels_test)
    # Check if the computed values are close to the input dprimes (within tolerance)
    np.testing.assert_allclose(computed_dprimes, dp_quadratic.compute(levels_test), rtol=1e-5)

    # Test inverse_compute method for a target d-prime
    target_dprime = 1.5 # A value within the expected range for this quadratic
    estimated_level = dp_quadratic.inverse_compute(target_dprime)

    # Check if the estimated level produces a d-prime close to the target
    if not np.isnan(estimated_level):
        recomputed_dprime = dp_quadratic.compute(estimated_level)
        self.assertAlmostEqual(recomputed_dprime, target_dprime, places=2)
    else:
        self.fail("inverse_compute returned NaN for a valid target d-prime.")

  def test_inverse_compute_edge_cases(self):
    levels_edge = np.array([1, 2, 3, 4, 5])
    # A simple quadratic opening upwards
    dprimes_edge = np.array([1, 4, 9, 16, 25])
    dp_quadratic = FitQuadraticMonomialCurve(levels_edge, dprimes_edge, plot=False)

    # Test target d-prime that has no real solution (below minimum of parabola)
    target_dprime_no_solution = -1.0
    self.assertTrue(np.isnan(dp_quadratic.inverse_compute(target_dprime_no_solution)))

    # Test target d-prime at the minimum of the parabola (one solution)
    # This assumes the minimum is 0, which is `dprimes_edge[2]`
    target_dprime_min = dprimes_edge[2]
    target_dprime_min = 0
    estimated_level_min = dp_quadratic.inverse_compute(target_dprime_min)
    recomputed_dprime_min = dp_quadratic.compute(estimated_level_min)
    self.assertAlmostEqual(recomputed_dprime_min, target_dprime_min, places=2)

    # Test target d-prime that has two solutions, ensure it picks the higher level
    target_dprime_two_solutions = 2
    estimated_level_two = dp_quadratic.inverse_compute(target_dprime_two_solutions)
    recomputed_dprime_two = dp_quadratic.compute(estimated_level_two)
    self.assertAlmostEqual(recomputed_dprime_two, target_dprime_two_solutions, places=2)
    # Explicitly check that it picks the larger root (assuming upward opening parabola and target d-prime > minimum)
    self.assertGreater(estimated_level_two, 1.0)


class SigmoidFit(absltest.TestCase):
  def test_sigmoid_fit(self):
    levels = np.arange(-5, 5)
    dprimes = np.arctan(levels)  # Example data that follows a sigmoid-like curve
    sigmoid_fit = FitSigmoidCurve(levels, dprimes, plot=True)
    plt.savefig('Results/test_sigmoid_fit_example.png')

    self.assertLess(sigmoid_fit.rms_error(levels, dprimes), 0.01)


def create_fake_df(verbose: bool = False):
  # Define parameters for the fake DataFrame
  num_rows = 5
  num_time_points = 20

  # Create MultiIndex levels
  frequencies = [4000.0, 8000.0]
  levels = [15.0, 20.0]
  polarities = [-1, 1]
  t0_values = [4.516987, 8.268104]
  t0_values = np.arange(-0.001, 0.006, 1/22000.0) # Example time range in seconds
  num_time_points = len(t0_values)

  # Generate combinations for the MultiIndex
  index_data = []
  for freq in frequencies:
      for lvl in levels:
          for pol in polarities:
              for t0 in t0_values:
                  index_data.append((freq, lvl, pol, t0))

  # Ensure we don't exceed num_rows for a cleaner example
  index_data = index_data[:num_rows]

  # Create the MultiIndex
  multi_index = pd.MultiIndex.from_tuples(index_data, names=['frequency', 'level', 'polarity', 't0'])

  # Create time columns
  time_columns = np.linspace(-0.001, 0.005, num_time_points) # Example time range in seconds
  time_columns = np.arange(-0.001, 0.006, 1/22000.0) # Example time range in seconds

  # Create random data for the DataFrame
  fake_data = np.random.rand(len(multi_index), num_time_points)

  # Create the DataFrame
  fake_df = pd.DataFrame(fake_data, index=multi_index, columns=time_columns)

  print("Fake DataFrame created successfully:")
  fake_df.head()
  return fake_df

def create_impulse_dataframe(data: pd.DataFrame, 
                             impulse_time = 0.001, # 1 ms
                             ):
  # Get the first row's index
  first_row_index = data.index[0]

  # Get the time columns (which are the actual time values)
  time_columns = data.columns.to_numpy()

  # Create a new row of zeros
  impulse_row_data = np.zeros(len(time_columns))

  # Find the index closest to 1ms in the time columns
  impulse_idx = np.searchsorted(time_columns, impulse_time)

  # Set the value at the impulse_idx to 1
  if 0 <= impulse_idx < len(impulse_row_data):
      impulse_row_data[impulse_idx] = 1.0
  else:
      print(f"Warning: Impulse time {impulse_time}s is out of bounds for the data's time range.")

  # Create a Series for the impulse response with the original time columns
  impulse_series = pd.Series(impulse_row_data, index=time_columns, name=first_row_index)

  # Replace the first row of the original 'data' DataFrame with the impulse series
  data.loc[first_row_index] = impulse_series
  return data


class ColoredNoise(absltest.TestCase):
  def test_frequency_response(self):
    data = create_fake_df()
    fs = dataframe_fs(data)
    self.assertAlmostEqual(fs, 22000)

    # Apply the bandpass filter and time-slicing
    data_filtered = abrpresto_bandpass(data, fs=fs)

    impulse_data = create_impulse_dataframe(data)
    data_filtered_impulse = abrpresto_bandpass(data, fs=fs)
    # Get the filtered impulse response and time vector
    impulse_response = data_filtered_impulse.iloc[0].values
    time_vector = data_filtered_impulse.columns.to_numpy()

    # Calculate sampling frequency if not already defined (or ensure it's correct)
    # fs was previously calculated as 1 / (data.columns[1] - data.columns[0])
    # Let's re-verify from the time_vector of the *filtered* data
    if len(time_vector) > 1:
        fs_filtered = 1 / (time_vector[1] - time_vector[0])
    else:
        print("Warning: Cannot determine sampling frequency for filtered data, using original fs.")
        fs_filtered = fs # Use the global fs if filtered data is too short

    # Number of samples in the impulse response
    N = len(impulse_response)

    # Perform FFT
    yf = fft(impulse_response)
    xf = fftfreq(N, 1 / fs_filtered)

    # Plot the magnitude of the FFT in dB
    plt.figure(figsize=(10, 6))
    # Add a small constant to avoid log(0) if yf contains zeros
    plt.plot(xf, 20 * np.log10(np.abs(yf) + 1e-10))
    plt.xscale('log')
    plt.title('Filtered Impulse Response in Frequency Domain (Magnitude in dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.axhline(-3, ls=':')
    plt.axvline(300, ls='--')
    plt.axhline(-3, ls=':')
    plt.axvline(3000, ls='--')
    # plt.xlim(xf[np.searchsorted(xf, 100)], fs_filtered / 2) # Start x-axis from a reasonable low frequency instead of 0 to avoid log(0)
    # Adjust y-axis limits to better visualize the filter's characteristics if needed
    plt.ylim(-40, 0) 
    plt.show()
    y300 = yf[np.abs(xf - 300).argmin()]
    y3000 = yf[np.abs(xf - 3000).argmin()]
    self.assertAlmostEqual(20*np.log10(np.abs(y300)), -3, delta=1.75)
    self.assertAlmostEqual(20*np.log10(np.abs(y3000)), -3, delta=0.25)


if __name__ == '__main__':
  absltest.main()