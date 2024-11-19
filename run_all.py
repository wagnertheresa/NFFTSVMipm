"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the all results presented in the paper in sequence.
"""

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
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)  # Ensure script runs successfully
    print(f"Finished {script}\n")
