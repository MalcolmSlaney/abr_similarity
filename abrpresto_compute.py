import multiprocessing as mp
from dataclasses import dataclass, asdict
import json
import pprint
import sys
from typing import Dict, List, Tuple
import time

import absl.flags as flags
import absl.app as app 

from abrpresto import compute_one_abrpresto_summary, combine_all_thresholds
from abrpresto import ABRSummary, get_threshold_data


# Define the error-handling wrapper function
def compute_wrapper(args: tuple) -> tuple:
    """
    Unpacks arguments, runs the routine, and catches any exceptions.
    Returns: (Success_Boolean, Payload)
    """
    try:
        # If it works, return True and the dataclass
        result = compute_one_abrpresto_summary(*args)
        return (True, result)
    except Exception as e:
        # If it fails, grab the identifying keys from the args to know WHAT failed
        mouse_id, timepoint, ear, frequency = args[:4]
        failed_key = (mouse_id, timepoint, ear, frequency)
        
        # Return False, the key, and the error message
        return (False, failed_key, str(e))


def load_results_from_json(json_path: str) -> Dict[str, Dict]:
  """Utility function to load results from a JSON file."""
  try:
    with open(json_path, 'r') as fp:
      results_dict = json.load(fp)
    assert isinstance(results_dict, dict), f"Expected a dictionary in the JSON file, got {type(results_dict)}"
    new_results_dict = {}
    for k, v in results_dict.items():
      k = k.split('_')
      k = (int(k[0]), int(k[1]), k[2], float(k[3]))  # Convert to tuple with correct types
      assert len(k) == 4, f"Expected key format 'mouse_id_timepoint_ear_frequency', got {k}"
      new_results_dict[k] = ABRSummary(**v)
    return new_results_dict

  except FileNotFoundError:
    print(f"No existing file found at {json_path}. Starting with an empty dictionary.")
  return {}


def filter_threshold_results(all_thresholds: List[tuple],  # All thresholds
                             results_dict: Dict[str, Dict] # Those already done
                             ) -> List[Tuple]:
  """Filters the list of all thresholds to only remove those that haven't been 
  processed yet."""
  assert isinstance(all_thresholds, list), f"Expected all_thresholds to be a list, got {type(all_thresholds)}"
  assert isinstance(results_dict, dict), f"Expected results_dict to be a dict, got {type(results_dict)}"
  filtered_thresholds = []
  skipped_rows = 0
  for row in all_thresholds:
    assert len(row) >= 4, f"Expected at least 4 elements in each row, got {len(row)}: {row}"
    key = (int(row[0]), int(row[1]), row[2], int(row[3]))  
    if key not in results_dict:
      filtered_thresholds.append(row)
    else:
      skipped_rows += 1
  print(f"Filtered thresholds: {len(filtered_thresholds)} remaining, {skipped_rows} already processed.")
  return filtered_thresholds

# basedir flag is defined in abrpresto.py, so it is aready defined and imported
flags.DEFINE_integer('max_rows', 0, 
                     'Number of rows in original manual thresholds file.')
flags.DEFINE_integer('num_workers', 0, 
                     'Number of worker processes to use for parallel processing.')
flags.DEFINE_string('output_path', 'Results/ABRPrestoSummary.json', 
                    'Path to save the output JSON file.')  
flags.DEFINE_float('cache_percent', 5, 
                   'How often to cache intermediate results to disk (as a percentage of total tasks).')
FLAGS = flags.FLAGS


# Main Execution
def main(argv):
  del argv # Unused.

  manual_df = get_threshold_data(FLAGS.basedir, 'Manual Thresholds.csv')
  abr_presto_df = get_threshold_data(FLAGS.basedir, 'ABRpresto thresholds 10-29-24.csv')
  all_thresholds = combine_all_thresholds(
      manual_df=manual_df,
      abrpresto_df=abr_presto_df
  )
  # Threshold data consists of: mouse_id, timepoint, ear, frequency,
  #                             manual_threshold, abrpresto_threshold
  if FLAGS.max_rows > 0:
    print(f"Limiting to the first {FLAGS.max_rows} rows of the manual thresholds.")
    all_thresholds = all_thresholds[:FLAGS.max_rows]

  if True:
    results_dict = load_results_from_json(FLAGS.output_path)
    all_thresholds = filter_threshold_results(all_thresholds, results_dict)
  # except Exception as e:
  #   print(f"Error loading existing results: {e}")
  #   print("Proceeding with all thresholds without filtering.")
  #   results_dict = {}

  # Job input data
  tasks = [(*row[:6], FLAGS.basedir, True) for row in all_thresholds]
  total_tasks = len(tasks)
  save_interval = max(1, int(total_tasks * FLAGS.cache_percent / 100))  # Save every X% of tasks
  num_workers = mp.cpu_count() - 1 or 1 
  num_workers = FLAGS.num_workers if FLAGS.num_workers > 0 else num_workers
  
  # List to keep track of failed tasks
  failed_tasks = []

  print(f"Starting processing with {num_workers} workers for {len(tasks)} tasks.")
  print(f"Will save partial results every {save_interval} tasks (5%).")

  with mp.Pool(processes=num_workers) as pool:
      result_iterator = pool.imap_unordered(compute_wrapper, tasks)
      
      # We wrap the iterator in enumerate(..., start=1) to count completed tasks
      for i, payload_tuple in enumerate(result_iterator, start=1):
          success, *payload = payload_tuple
          
          if success:
              summary = payload[0]
              if summary is None:
                  continue  # Returns none if the data isn't available on this machine.
              key = str(summary.mouse_id) + "_" + str(summary.timepoint) + "_" + str(summary.ear) + "_" + str(summary.frequency)
              results_dict[key] = asdict(summary)
              print(f'Successfully processed {i}: {key}')
          else:
              failed_key, error_msg = payload
              failed_tasks.append((failed_key, error_msg))
              print(f"[!] Task failed for {failed_key}: {error_msg}")


          # --- Check if we hit the 5% interval threshold ---
          if i % save_interval == 0:
              print(f"--> Checkpoint: Saving partial results ({i}/{total_tasks} complete)...")
              with open(FLAGS.output_path, "w") as fp:
                  json.dump(results_dict, fp, indent=4)
          sys.stdout.flush()  # Flush status messages so far

  # Final Output Summary
  print("\n--- Processing Complete ---")
  print(f"Successfully processed: {len(results_dict)}")
  print(f"Failed tasks: {len(failed_tasks)}")
  print('Payload ends with:', payload)
  
  # You can now safely access your populated dictionary
  with open(FLAGS.output_path, 'w') as fp:
    json.dump(results_dict, fp, indent=4)

if __name__ == "__main__":
    app.run(main)