"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the all results presented in the paper in sequence.
"""
################
# dynamically determine location if test.py and adjust paths accordingly
import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path to allow importing the run files
sys.path.insert(0, script_dir)

#################
import subprocess

# List of all scripts to execute
scripts = [
    "run_test_precond_timings.py",
    "run_test_IPMGMRES_iters.py",
    "run_test_IPMGMRES_tols.py",
    "run_test_dratio.py",
    "run_finaltest.py"
]

# Loop through each script and execute it
for script in scripts:
    # Construct the absolute path to the specific file in the test folder
    file_path = os.path.join(script_dir, script)
    print(f"Running {script}...")
    subprocess.run(["python", file_path,'--Ndata', '50000'], check=True)  # Ensure script runs successfully
    print(f"Finished {script}\n")
