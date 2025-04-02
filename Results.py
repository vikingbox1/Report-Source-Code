#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:29:28 2025

@author: johnpainuvila
"""

import numpy as np
from CliffordOps import *
from NHow import *
import math
from numpy.linalg import matrix_rank

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

######################################################
## Any code with at least 1 logical qubit
######################################################

## Logical S
#U = symKron([symCZ(1,0,0),ZMatI(2*(k-1))])

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

def stabiliserChecker(LC, S):
    
    TS = matMul(S, LC, 2) #Transformed stabiliser group
    
    
    stand_TS = getH(TS,2)
    stand_S = getH(S,2)
    
    if np.all(stand_S == stand_TS):
        return True
    
    else:
        return False
    
def vXZ(vList):
    q = int(len(vList[0])/2)
    print("qubits:",q)
    print("transvections:",len(vList))
    for i in range(len(vList)):
        v = vList[i]
        xList = v[:q]
        zList = v[q:]
        
        xInd =  np.where(xList == 1)[0]
        zInd =  np.where(zList == 1)[0]
        
        xStr = ""
        if len(xInd) > 0:
            for n in range(len(xInd)):
                xStr += " " + f"X{xInd[n]}"
                
        zStr = ""
        if len(zInd) > 0:
            for n in range(len(zInd)):
                zStr += " " + f"Z{zInd[n]}"
                
        print(f"v{i}:", xStr, zStr)
    
def clifford_layer_to_gates(tableau):
    """
    Given a binary symplectic tableau for a layer of single-qubit Clifford operations,
    return a list of gate names for each qubit.
    
    The input 'tableau' should be a numpy array of shape (2*n, 2*n), where the first n rows
    represent the images of the X generators and the last n rows the images of the Z generators.
    The left half (columns 0 to n-1) corresponds to the X part and the right half (columns n to 2*n-1)
    to the Z part.
    
    For each qubit i (0-indexed), we extract the 2x2 binary matrix:
      M = [[tableau[i, i],        tableau[i, n+i]],
           [tableau[n+i, i], tableau[n+i, n+i]]]
    and then match it to one of the 6 possible single-qubit Clifford operations (modulo phase).
    
    The dictionary below maps the 2x2 binary matrices (represented as a tuple
    (a, b, c, d)) to a canonical gate name.
    
    Returns:
        List of gate names (strings) corresponding to each qubit.
    """
    # Define the mapping from 2x2 binary matrix (flattened as tuple) to gate name.
    mapping = {
        (1, 0, 0, 1): "I",
        (1, 1, 0, 1): "S",
        (1, 1, 1, 0): "SH",
        (0, 1, 1, 0): "H",
        (0, 1, 1, 1): "HS",
        (1, 0, 1, 1): "HSH",
        (0, 1, 0, 1): "HSHS",       # X -> Z, Z -> Y (approximate label)
        (1, 1, 1, 1): "S^2H",
        (1, 0, 1, 0): "H^2",        # 180 degree rotation (H^2 = XZ)
        (0, 0, 1, 1): "HS^2",       # etc.
        (1, 1, 0, 0): "S^2",
        (0, 0, 1, 0): "X",
        (0, 0, 0, 1): "Z",
        (1, 0, 0, 0): "Y",
        (1, 1, 1, 1): "S^2H",       # Duplicates may exist if relabeling is ambiguous
        (1, 1, 0, 1): "S",
        (0, 1, 0, 0): "HZ",
        (0, 0, 1, 1): "HS^2",
        (1, 0, 1, 0): "XZ",
        (0, 1, 1, 0): "H",
        (1, 0, 1, 1): "HSH",
        (1, 1, 1, 0): "SH",
        (0, 1, 1, 1): "HS",
        (0, 0, 0, 0): "???",  # Invalid symplectic matrix
    }
    
    # Determine the number of qubits.
    num_rows, num_cols = tableau.shape
    if num_rows != num_cols or num_rows % 2 != 0:
        raise ValueError("Tableau must be a square matrix with even dimensions (2*n x 2*n).")
    n = num_rows // 2
    
    gates = []
    for i in range(n):
        # Extract the 2x2 block corresponding to qubit i.
        a = int(tableau[i, i])
        b = int(tableau[i, n+i])
        c = int(tableau[n+i, i])
        d = int(tableau[n+i, n+i])
        key = (a, b, c, d)
        gate = mapping.get(key, None)
        if gate is None:
            raise ValueError(f"Unrecognized single-qubit Clifford operation for qubit {i}: {key}")
        gates.append(gate)
    
    return gates


#S
# Load the matrix (example: Us from simulation 0)
Us = np.load("mkiiToricS_matrices/Us_017.npy")
C = np.load("mkiiToricS_matrices/C_017.npy")
A = np.load("mkiiToricS_matrices/A_017.npy")

# Print the matrix
print("Us matrix:\n", Us)
print("C matrix:\n", C)
print("A matrix:\n", A)


print("check to see if stabiliser group has been preserved")   
print("MKII:", stabiliserChecker(Us, S0))


## Decompose into 2-qubit transvections
vList,ixC,UC = transDecomp(Us)
print('vList',vList)
print('ixC',ixC)
print(clifford_layer_to_gates(UC))
#print('UC')
#print(ZMatPrint(UC,tB=2))
#print('Original Tableau')
#print(ZMatPrint(T,tB=2))
#T1 = matMul(T,Us,2)
#print('Transformed Tableau')
#print(ZMatPrint(T1,tB=2))





#H
Us = np.load("mkiiToricH2_matrices/Us_035.npy")
C = np.load("mkiiToricH2_matrices/C_035.npy")
A = np.load("mkiiToricH2_matrices/A_035.npy")

# Print the matrix
print("Us matrix:\n", Us)
print("C matrix:\n", C)
print("A matrix:\n", A)


print("check to see if stabiliser group has been preserved")   
print("MKII:", stabiliserChecker(Us, S0))


## Decompose into 2-qubit transvections
vList,ixC,UC = transDecomp(Us)
print('vList',vList)
print('ixC',ixC)
print(clifford_layer_to_gates(UC))
#print('UC')
#print(ZMatPrint(UC,tB=2))



#CZ
Us = np.load("mkiiToricCZ2_matrices/Us_008.npy")
C = np.load("mkiiToricCZ2_matrices/C_008.npy")
A = np.load("mkiiToricCZ2_matrices/A_008.npy")

# Print the matrix
print("Us matrix:\n", Us)
print("C matrix:\n", C)
print("A matrix:\n", A)


print("check to see if stabiliser group has been preserved")   
print("MKII:", stabiliserChecker(Us, S0))


## Decompose into 2-qubit transvections
vList,ixC,UC = transDecomp(Us)
print('vList',vList)
print('ixC',ixC)
vXZ(vList)
print(clifford_layer_to_gates(UC))
#print('UC')
#print(ZMatPrint(UC,tB=2))




#CNOT
Us = np.load("mkiiToricCNOT_matrices/Us_004.npy")
C = np.load("mkiiToricCNOT_matrices/C_004.npy")
A = np.load("mkiiToricCNOT_matrices/A_004.npy")

# Print the matrix
print("Us matrix:\n", Us)
print("C matrix:\n", C)
print("A matrix:\n", A)


print("check to see if stabiliser group has been preserved")   
print("MKII:", stabiliserChecker(Us, S0))


## Decompose into 2-qubit transvections
vList,ixC,UC = transDecomp(Us)
print('vList',vList)
print('ixC',ixC)
vXZ(vList)
print(clifford_layer_to_gates(UC))
#print('UC')
#print(ZMatPrint(UC,tB=2))
