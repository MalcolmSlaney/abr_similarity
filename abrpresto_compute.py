
import multiprocessing as mp
from dataclasses import dataclass, asdict
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

import absl.flags as flags
import absl.app as app 

from abrpresto import compute_one_abrpresto_summary, get_threshold_data
from abrpresto import ABRSummary, get_all_manual_thresholds

# 1. Define your dataclass
@dataclass
class XXABRSummary:
    mouse_id: str
    timepoint: str
    ear: str
    frequency: float
    computed_value: float 


# 3. Define the error-handling wrapper function
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


# basedir flag is defined in abrpresto.py, so it is aready defined and imported.c
flags.DEFINE_integer('max_rows', 0, 
                     'Number of rows in original manual thresholds file.')
flags.DEFINE_integer('num_workers', 0, 
                     'Number of worker processes to use for parallel processing.')
flags.DEFINE_string('output_path', 'Results/ABRPrestoSummary.json', 
                    'Path to save the output JSON file.')  
FLAGS = flags.FLAGS


# 4. Main Execution
def main(argv):
  del argv # Unused.

  manual_df = get_threshold_data(FLAGS.basedir, 'Manual Thresholds.csv')
  abr_presto_df = get_threshold_data(FLAGS.basedir, 'ABRpresto thresholds 10-29-24.csv')
  all_manual_thresholds = get_all_manual_thresholds(
      manual_df=manual_df,
      abrpresto_df=abr_presto_df
  )
  all_manual_thresholds = all_manual_thresholds[:FLAGS.max_rows] if FLAGS.max_rows > 0 else all_manual_thresholds

  # Job input data
  tasks = [(*row[:6], FLAGS.basedir, True) for row in all_manual_thresholds]
  num_workers = mp.cpu_count() - 1 or 1 
  num_workers = FLAGS.num_workers if FLAGS.num_workers > 0 else num_workers
  
  # Dictionary for successful results
  results_dict: Dict[Tuple[str, str, str, float], ABRSummary] = {}
  
  # List to keep track of failed tasks
  failed_tasks = []

  print(f"Starting processing with {num_workers} workers for {len(tasks)} tasks.")

  with mp.Pool(processes=num_workers) as pool:
      result_iterator = pool.imap_unordered(compute_wrapper, tasks)
      
      for success, *payload in tqdm(result_iterator, total=len(tasks), desc="Processing ABRs"):
          if success:
              # Unpack the successful payload (the dataclass)
              summary = payload[0]
              if summary is None:
                  continue  # Returns none if the data isn't available on this machine.
              key = str(summary.mouse_id) + "_" + str(summary.timepoint) + "_" + str(summary.ear) + "_" + str(summary.frequency)
              results_dict[key] = asdict(summary)
          else:
              # Unpack the failed payload (key and error message)
              failed_key, error_msg = payload
              failed_tasks.append((failed_key, error_msg))
              # Using tqdm.write instead of print prevents the progress bar from breaking visually
              tqdm.write(f"[!] Task failed for {failed_key}: {error_msg}")

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