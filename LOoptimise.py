import numpy as np
from CliffordOps import *
from NHow import *

def OptLO(T,U,settgs=None,seed=0):
    '''For tableau T and desired logical Clifford operator U find logical U using minimal 2-qubit transvections'''
    startTimer()
    Tinv = symInverse(T) 
    k = len(U)//2
    n = len(T)//2
    r = n-k
    rnd = np.random.default_rng(seed)
    time = 0
    defs = {'lambmu':10,'mu':n ** 4,'genCount':(n) ** 2,'k':k,'n':n,'r':r,'nilA1':False}
    ## use defaults for any settings not in settgs
    if settgs is not None:
        settgs = defs | settgs
    else:
        settgs = defs
    nilA1 = settgs['nilA1']
    nPop = settgs['lambmu']*settgs['mu']
    ## random initial population
    population = [bin2CA(randomCAbin(rnd,n,k),n,k) for i in range(nPop)]
    ## evaluate initial population
    LOScores = [OptLOEval(T,Tinv,U,C,A,nilA1) for (C,A) in population]
    ## find best solution so far
    j = argmin(LOScores)
    bestScore = LOScores[j]
    bestAC = population[j]
    time = time + elapsedTime()
    print(func_name(),bestScore,time)
    C,A = population[0]
    for g in range(settgs['genCount']):
        print(g)
        ix = argsort(LOScores,reverse=False)[:settgs['mu']]
        ## mutate to form next generation
        population = [OptLOMutate(population[j],settgs,rnd) for i in range(settgs['lambmu']) for j in ix]
        ## evaluate population
        LOScores = [OptLOEval(T,Tinv,U,C,A,nilA1) for (C,A) in population]
        ## find best solution so far
        j = argmin(LOScores)
        # print(func_name(),j, len(LOScores))
        
        if LOScores[j] < bestScore:
            bestScore = LOScores[j]
            bestAC = population[j]
            time = time + elapsedTime()
            print(func_name(),bestScore,time)
            # C,A = bestAC
            # Us = matMul(Tinv,matMul(UCA2sym(U,C,A,nilA1),T,2),2)
            # print(ZMatPrint(Us,tB=2))
        if np.sum(bestScore) == 0:
            break

    return bestScore,bestAC, time + elapsedTime()

def OptLOEval(T,Tinv,U,C,A,nilA1):
    ## make logical operator from U,C,A
    n,r = C.shape
    if nilA1:
        A[:r,:] = 0
    Us = matMul(Tinv,matMul(UCA2sym(U,C,A),T,2),2)
    ## Decompose into 2-qubit transvections
    vList,ixC,UC = transDecomp(Us)
    return transScore(vList,T)

def OptLOMutate(x,settgs,rng):
    r = settgs['r']
    k = settgs['k']
    C,A = x
    if settgs['nilA1']:
        A[:r,:] = 0
    C = C.copy()
    C1,C2 = C[:r],C[r:]
    AC = ZMatVstack([A,C2])
    nC1 = r * (r-1)
    ACoff = (r if settgs['nilA1'] else 0)
    nAC = r * (r - ACoff + 2 * k )
    i = rng.integers(nC1 + nAC )
    if i < nC1:
        ## row operation on C
        a = i % r
        b = (a + (i // r + 1)) % r
        # print(C.shape, a,b)
        C1[b] ^= C1[a]
    else:
        ## flip bit of AC matrix
        i = i - nC1 + (r * ACoff)              
        a = i // r
        b = i % r
        # print(A.shape,a,b)
        AC[a,b] ^= 1
    A = AC[:r + k]
    C = ZMatVstack([C1,AC[r+k:]])
    return C,A

