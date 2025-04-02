#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:56:25 2025

@author: johnpainuvila
"""

###################################################
#lambda ≈ 1/(n-1) * (r * (r-1) + r * (r + 2 * k )), 
#proportions -> tour size ≈ lambda
#
#
###################################################



from M1 import *
from M2 import *
from MKII import *



import concurrent.futures
import numpy as np

## Codetables formatted Codes
#mycode = '3-1-1'
# mycode = '4-2-2'
mycode = '5-1-3'
# mycode = '7-1-3'
# mycode = '8-3-3'
# mycode = '9-1-3'
# mycode = '11-2-3'
# mycode = '12-2-4'
# mycode = '12-6-3'
# mycode = '18-10-2'


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
U = symKron([symCZ(1,0,0),ZMatI(2*(k-1))])

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
# U = symCZ(k,0,1)

## Logical CNOT
# U = symCNOT(k,0,1)

Tinv = symInverse(T)

import numpy as np
import concurrent.futures
import pandas as pd
import os


def findLO(_):
    # Generate a seed for this iteration.
    seeds = np.random.SeedSequence().generate_state(1)
    seed = seeds[0]

    # Prepare dictionaries to hold results
    scores = {}
    times = {}
    trace_log = {}
    
    # Run algorithms    
    mkii, (_,_), t_mkii, l_mkii = MKIIOptLO(T, U, seed=seed)

    # Store scores
    scores["mkii"] = np.sum(mkii)


    # Store times
    times["t_mkii"] = t_mkii


    # Store trace logs
    trace_log["l_mkii"] = l_mkii
    

    return scores, times, trace_log 



if __name__ == '__main__':
    # Define the number of simulation iterations.
    num_simulations = 500
    
    # File names
    scores_file = "mkiic5LS_scores.csv"
    times_file = "mkiic5LS_times.csv"
    trace_file = "mkiic5LS_trace.csv"

    # Check if files exist (to determine if we need headers)
    scores_exists = os.path.isfile(scores_file)
    times_exists = os.path.isfile(times_file)
    trace_exists = os.path.isfile(trace_file)

    # Define columns for scores
    score_columns = [
        "mkii"
    ]
    
    # Define columns for times
    time_columns = [
        "t_mkii"
    ]
    
    # Define columns for trace logs
    trace_columns = [
        "l_mkii"
    ]
    


    
    # Run the simulation using a process pool.
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(findLO, i): i for i in range(num_simulations)}
        
        for future in concurrent.futures.as_completed(futures):
            scores, times, trace_log = future.result()

            # Convert to DataFrames
            scores_df = pd.DataFrame([scores])
            times_df = pd.DataFrame([times])
            trace_df = pd.DataFrame([trace_log])

            # Save scores to file
            scores_df.to_csv(scores_file, mode='a', header=not scores_exists, index=False)
            scores_exists = True  # Ensure header is only written once

            # Save times to file
            times_df.to_csv(times_file, mode='a', header=not times_exists, index=False)
            times_exists = True  # Ensure header is only written once
            
            trace_df.to_csv(trace_file, mode='a', header=not trace_exists, index=False)
            trace_exists = True  # Ensure header is only written once

            print(f"Scores: {scores}, Times: {times}")  # Optionally print each result