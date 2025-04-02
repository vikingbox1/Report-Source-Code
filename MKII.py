#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 12:20:26 2025

@author: johnpainuvila
"""

import numpy as np
from CliffordOps import *
from NHow import *

def tournament_selection_subset(LOScores, candidate_indices, tournament_size, num_to_select, rnd):
    """
    Select num_to_select candidate indices from candidate_indices using tournament selection.
    LOScores: list of candidate scores.
    candidate_indices: list/array of indices from which to select.
    tournament_size: number of individuals to sample in each tournament.
    rnd: random generator.
    """
    selected_indices = []
    candidate_array = np.array(candidate_indices)
    for _ in range(num_to_select):
        ts = tournament_size if tournament_size <= len(candidate_array) else len(candidate_array)
        participants = rnd.choice(candidate_array, size=ts, replace=False)
        best_index = min(participants, key=lambda idx: LOScores[idx])
        selected_indices.append(best_index)
    return selected_indices

def MKIIOptLO(T, U, settgs=None, seed=0):
    '''For tableau T and desired logical Clifford operator U find logical U using minimal 2-qubit transvections'''
    startTimer()
    Tinv = symInverse(T) 
    k = len(U) // 2
    n = len(T) // 2
    r = n - k
    rnd = np.random.default_rng(seed)
    time = 0
    defs = {
        'lambmu': int((1/(n-1)) * (r * (r-1) + r * (r + 2 * k )) + 1),
        'mu': n ** 4,
        'genCount': (n+1) ** 2,
        'k': k,
        'n': n,
        'r': r,
        'nilA1': False,
        'elite_fraction': 0.5,
    }
    ## use defaults for any settings not in settgs
    if settgs is not None:
        settgs = defs | settgs
    else:
        settgs = defs
        
    nilA1 = settgs['nilA1']
    nPop = settgs['lambmu'] * settgs['mu']
    mu = settgs['mu']
    elite_fraction = settgs['elite_fraction']
    tournament_size = settgs['lambmu'] - 1

    ## random initial population
    population = [bin2CA(randomCAbin(rnd, n, k), n, k) for i in range(nPop)]
    ## evaluate initial population
    LOScores = [OptLOEval(T, Tinv, U, C, A, nilA1) for (C, A) in population]
    ## find best solution so far
    j = argmin(LOScores)
    bestScore = LOScores[j]
    bestAC = population[j]
    time = time + elapsedTime()
    print(func_name(), bestScore, time)
    
    # Initialize stall detection variables.
    fertility = 0
    stall_count = 0
    
    trace_log = []
    
    for g in range(settgs['genCount']):
        print(g)
        
        elite_count = int(elite_fraction * mu)
        random_count = mu - elite_count
        
        sorted_indices = argsort(LOScores, reverse=False)
        elite_indices = sorted_indices[:elite_count]
        remaining_indices = sorted_indices[elite_count:]
        
        # Replace random selection with tournament selection on remaining indices.
        if random_count > len(remaining_indices):
            tournament_indices = remaining_indices
        else:
            tournament_indices = tournament_selection_subset(LOScores, remaining_indices, tournament_size, random_count, rnd)
        
        ix = np.concatenate([elite_indices, tournament_indices])
        
        if stall_count == int(1.5*n):
            mutation_strength = n
        else:
            mutation_strength = 1
        
        ## Mutate to form next generation using the chosen mutation strength.
        population = [OptLOMutate(population[j], settgs, rnd, m=mutation_strength)
                      for i in range(settgs['lambmu'] + fertility) for j in ix]
                      
        ## Evaluate new population.
        LOScores = [OptLOEval(T, Tinv, U, C, A, nilA1) for (C, A) in population]
        ## Find best solution so far.
        j = argmin(LOScores)
        
        if sum(LOScores[j]) < sum(bestScore):
            bestScore = LOScores[j]
            bestAC = population[j]
            time = time + elapsedTime()
            print(func_name(), bestScore, time)
            fertility = 0
            stall_count = 0
        elif LOScores[j] < bestScore:
            fertility = 0
            stall_count = 0
            bestScore = LOScores[j]
            bestAC = population[j]
            time = time + elapsedTime()
            print(func_name(), bestScore, time)
        else:
            stall_count += 1
            fertility += 1
            fertility = min(fertility, int(n/2))
            
        # Log the current state of the desired variables.
        trace_log.append({
            "g": g,
            "elite_fraction": elite_fraction,
            "tournament_size": tournament_size,
            "best_LOScore": sum(LOScores[j]),
            "mutation_strength": mutation_strength
        })
            
        # Early termination if a perfect candidate is found.
        if np.sum(bestScore) == 0:
            break

    return bestScore, bestAC, time + elapsedTime(), trace_log

def OptLOEval(T, Tinv, U, C, A, nilA1):
    ## Make logical operator from U, C, A.
    n, r = C.shape
    if nilA1:
        A[:r, :] = 0
    Us = matMul(Tinv, matMul(UCA2sym(U, C, A), T, 2), 2)
    ## Decompose into 2-qubit transvections.
    vList, ixC, UC = transDecomp(Us)
    return transScore(vList, T)

def OptLOMutate(x, settgs, rng, m=1):
    """
    Apply m single mutation operations on candidate (C, A).
    Each mutation is either a row operation on C or a bit flip in the concatenated matrix [A; C2].
    """
    r = settgs['r']
    k = settgs['k']
    C, A = x
    if settgs['nilA1']:
        A[:r, :] = 0
    C = C.copy()
    C1, C2 = C[:r], C[r:]
    AC = ZMatVstack([A, C2])
    nC1 = r * (r - 1)
    ACoff = (r if settgs['nilA1'] else 0)
    nAC = r * (r - ACoff + 2 * k)
    for _ in range(m):
        i = rng.integers(nC1 + nAC)
        if i < nC1:
            # Perform a row operation on C.
            a = i % r
            b = (a + (i // r + 1)) % r
            C1[b] ^= C1[a]
        else:
            # Flip a bit in the AC matrix.
            i_adj = i - nC1 + (r * ACoff)
            a = i_adj // r
            b = i_adj % r
            AC[a, b] ^= 1
    A = AC[:r + k]
    C = ZMatVstack([C1, AC[r + k:]])
    return C, A


###################################################
###################################################
#
###################################################
###################################################

#checks to see if the logical operator preserves the stabiliser generators up to some permutation
def stabiliserChecker(LC, S):
    
    TS = matMul(S, LC, 2) #Transformed stabiliser group
    
    
    stand_TS = getH(TS,2)
    stand_S = getH(S,2)
    
    if np.all(stand_S == stand_TS):
        return True
    
    else:
        return False



## Codetables formatted Codes
# mycode = '3-1-1'
# mycode = '4-2-2'
mycode = '5-1-3'
#mycode = '7-1-3'
# mycode = '8-3-3'
# mycode = '9-1-3'
# mycode = '11-2-3'
# mycode = '12-2-4'
# mycode = '12-6-3'
# mycode = '18-10-2'
# mycode = '8-2-2Toric'


#print(f'Clifford LO {mycode}')
f = open(f'codeTables/{mycode}.txt')
mystr = f.read()
f.close()

## Calculate tableau form
mystr = mystr.replace(" ","").replace("[","").replace("]","").replace("|","")
S0 = bin2ZMat(mystr.strip())

#print('S0')
#print(ZMatPrint(S0,tB=2))

n,k,pT,T = Stab2Tableau(S0)

seeds = np.random.SeedSequence().generate_state(1)
seed = seeds[0]

######################################################
## Any code with at least 1 logical qubit
######################################################

## Logical S
# U = symKron([symCZ(1,0,0),ZMatI(2*(k-1))])

## Logical sqrt X
# U = symKron([symCXX(1,0,0),ZMatI(2*(k-1))])

## Logical H
U = symKron([OmegaMat(1),ZMatI(2*(k-1))])

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

#print('U')
#print(ZMatPrint(U,tB=2))

#bestScore, (C,A), time, trace_log = MKIIOptLO(T,U,seed=seed)

def stabiliserChecker(LC, S):
    
    TS = matMul(S, LC, 2) #Transformed stabiliser group
    
    
    stand_TS = getH(TS,2)
    stand_S = getH(S,2)
    
    if np.all(stand_S == stand_TS):
        return True
    
    else:
        return False
    
## make logical operator from U,C,A
#Us = matMul(Tinv,matMul(UCA2sym(U,C,A),T,2),2)
#print("check to see if stabiliser group has been preserved")   
#print("MKII:", stabiliserChecker(Us, S0))

###############################################
###############################################
#
###############################################
###############################################

import matplotlib.pyplot as plt
import numpy as np

# Assume trace_log is your list of dictionaries containing the logged data.
#g = [entry["g"] for entry in trace_log]
#best_LOScore = [entry["best_LOScore"] for entry in trace_log]
#elite_fraction = [entry["elite_fraction"] for entry in trace_log]
#fertility = [entry["fertility"] for entry in trace_log]
#tournament_size = [entry["tournament_size"] for entry in trace_log]



def plot_dual_y(g, variable, var_name, best_LOScore):
    fig, ax1 = plt.subplots()

    # Primary y-axis for the variable.
    color_var = 'blue'
    ax1.plot(g, variable, marker='o', color=color_var, label=var_name)
    ax1.set_xlabel("g")
    ax1.set_ylabel(var_name, color=color_var)
    ax1.tick_params(axis='y', labelcolor=color_var)

    # Secondary y-axis for best_LOScore.
    ax2 = ax1.twinx()
    color_best = 'red'
    ax2.plot(g, best_LOScore, marker='x', linestyle='--', color=color_best, label='best_LOScore')
    ax2.set_ylabel("best_LOScore", color=color_best)
    ax2.tick_params(axis='y', labelcolor=color_best)

    fig.tight_layout()
    plt.title(f"{var_name} and best_LOScore vs g")
    plt.show()

#plot_dual_y(g, elite_fraction, "elite_fraction", best_LOScore)

#plot_dual_y(g, tournament_size, "tournament_size", best_LOScore)

#plot_dual_y(g, fertility, "fertility", best_LOScore)