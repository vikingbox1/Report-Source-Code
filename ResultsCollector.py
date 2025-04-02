#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:03:42 2025

@author: johnpainuvila
"""

from MKII import *


import concurrent.futures
import numpy as np

## Codetables formatted Codes
# mycode = '3-1-1'
# mycode = '4-2-2'
# mycode = '5-1-3'
# mycode = '7-1-3'
# mycode = '8-3-3'
# mycode = '9-1-3'
# mycode = '11-2-3'
# mycode = '12-2-4'
# mycode = '12-6-3'
# mycode = '18-10-2'
mycode = '8-2-2Toric'

f = open(f'codeTables/{mycode}.txt')
mystr = f.read()
f.close()

## Calculate tableau form
mystr = mystr.replace(" ","").replace("[","").replace("]","").replace("|","")
S0 = bin2ZMat(mystr.strip())

n,k,pT,T = Stab2Tableau(S0)


######################################################
## Any code with at least 1 logical qubit
######################################################

## Logical S
# U = symKron([symCZ(1,0,0),ZMatI(2*(k-1))])

## Logical sqrt X
# U = symKron([symCXX(1,0,0),ZMatI(2*(k-1))])

## Logical H
#U = symKron([OmegaMat(1),ZMatI(2*(k-1))])

######################################################
## For codes with at least 2 logical qubits only!
######################################################

## Logical S1S2
# U = symKron([symCZ(1,0,0),symCZ(1,0,0)])

## Logical H1H2
# U = symKron([OmegaMat(2),ZMatI(2*(k-2))])

## Logical CZ
U = symCZ(k,0,1)

## Logical CNOT
# U = symCNOT(k,0,1)

Tinv = symInverse(T)

import numpy as np
import concurrent.futures
import pandas as pd
import os


def findLO(sim_id):
    # Generate a seed for this iteration.
    seeds = np.random.SeedSequence().generate_state(1)
    seed = seeds[0]

    # Prepare dictionaries to hold results
    scores = {}
    times = {}
    trace_log = {}

    # Run algorithm    
    mkii, (C, A), t_mkii, l_mkii = MKIIOptLO(T, U, seed=seed)

    # Store scores
    scores["mkii"] = np.sum(mkii)

    # Store times
    times["t_mkii"] = t_mkii

    # Store trace logs
    trace_log["l_mkii"] = l_mkii

    # Compute Us matrix
    Us = matMul(Tinv, matMul(UCA2sym(U, C, A), T, 2), 2)

    return scores, times, trace_log, Us, C, A, sim_id






if __name__ == '__main__':
    import numpy as np

    # Define the number of simulation iterations.
    num_simulations = 100

    # File names
    scores_file = "mkiiToricCZ2_scores.csv"
    times_file = "mkiiToricCZ2_times.csv"
    trace_file = "mkiiToricCZ2_trace.csv"
    matrix_dir = "mkiiToricCZ2_matrices"

    # Create a directory to store matrices if it doesn't exist
    os.makedirs(matrix_dir, exist_ok=True)

    # Check if files exist
    scores_exists = os.path.isfile(scores_file)
    times_exists = os.path.isfile(times_file)
    trace_exists = os.path.isfile(trace_file)

    # Define columns
    score_columns = ["mkii"]
    time_columns = ["t_mkii"]
    trace_columns = ["l_mkii"]

    sim_ids_file = "mkiiToricCZ2_sim_ids.csv"
    sim_ids_exists = os.path.isfile(sim_ids_file)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(findLO, i): i for i in range(num_simulations)}

        for future in concurrent.futures.as_completed(futures):
            scores, times, trace_log, Us, C, A, sim_id = future.result()

            # Convert to DataFrames
            scores_df = pd.DataFrame([scores])
            times_df = pd.DataFrame([times])
            trace_df = pd.DataFrame([trace_log])

            # Save CSVs
            scores_df.to_csv(scores_file, mode='a', header=not scores_exists, index=False)
            scores_exists = True

            times_df.to_csv(times_file, mode='a', header=not times_exists, index=False)
            times_exists = True

            trace_df.to_csv(trace_file, mode='a', header=not trace_exists, index=False)
            trace_exists = True

            # Save matrices
            np.save(os.path.join(matrix_dir, f"Us_{sim_id:03d}.npy"), Us)
            np.save(os.path.join(matrix_dir, f"C_{sim_id:03d}.npy"), C)
            np.save(os.path.join(matrix_dir, f"A_{sim_id:03d}.npy"), A)

            # Save sim_id to file in return order
            with open(sim_ids_file, 'a') as f:
                if not sim_ids_exists:
                    f.write("sim_id\n")
                    sim_ids_exists = True
                f.write(f"{sim_id}\n")

            print(f"Scores: {scores}, Times: {times}, SimID: {sim_id}")



