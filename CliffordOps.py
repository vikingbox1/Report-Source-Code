import numpy as np
from NHow import *

##########################################################################
## Basics of Symplectic Matrices
##########################################################################

def OmegaMat(n):
    '''Omega is the Symplectic form on 2n qubits
    Also Hadamard on all n qubits'''
    E = np.array([[0,1],[1,0]],dtype=int)
    In = np.eye(n,dtype=int)
    return np.kron(E,In)

def permMat(perm):
    '''take permutation vector and convert to perm matrix'''
    n = len(perm)
    temp = np.zeros((n,n),dtype=int)
    for i in range(n):
        temp[perm[i],i] = 1
    return temp

def isSymplectic(S):
    '''check if matrix S is binary symplectic'''
    n = len(S)//2
    Om = OmegaMat(n)
    return 0 == np.sum(matMul(S.T, matMul(Om,S,2),2) ^ Om)
    return 0 == np.sum(np.mod(S.T @ Om @ S - Om, 2))

def symInverse(S):
    '''Inverse of Sympletic matrix S'''
    n = len(S)//2
    On = OmegaMat(n)
    return matMul(On,matMul(S.T,On,2),2)

def transvection(v):
    '''Construct symplectic matrix Tv correponding to transvection of vector v'''
    n = len(v)//2
    Ov = ZMatZeros(2*n)
    Ov[:n] = v[n:]
    Ov[n:] = v[:n]
    OvT = ZMat2D(Ov).T
    Tv = matMul(OvT,v,2)
    for i in range(2*n):
        Tv[i,i] ^= 1
    return Tv

## Symplectic form of well-known Clifford operators 
def symCNOT(n,i,j):
    '''CNOT_ij on n qubits as symplectic matrix'''
    C = ZMatI(n)
    C[j] ^= C[i]
    S = ZMatZeros((2*n,2*n))
    S[:n,:n] = C 
    S[n:,n:] = C.T
    return S

def symCZ(n,i,j):
    '''CZ_ij on n qubits as symplectic matrix
    CZ_ii = S_i'''
    Q = ZMatZeros((n,n))
    Q[i,j] = 1
    Q[j,i] = 1
    S = ZMatI(2*n)
    S[:n,n:] = Q
    return S

def symCXX(n,i,j):
    '''C(X,X)_ij on n qubits as symplectic matrix
    C(X,X)_ii = sqrt{X}_i'''
    return symCZ(n,i,j).T

def symKron(SList):
    '''Kronecker product of list of symplectic matrices'''
    nList = [len(S)//2 for S in SList]
    n = np.sum(nList)
    S = ZMatZeros((2*n,2*n))
    c = 0
    for ni,Si in zip(nList,SList):
        S[c:c+ni,c:c+ni] = Si[:ni,:ni]
        S[c:c+ni,n+c:n+c+ni] = Si[:ni,ni:]
        S[n+c:n+c+ni,c:c+ni] = Si[ni:,:ni]
        S[n+c:n+c+ni,n+c:n+c+ni] = Si[ni:,ni:]
        c += ni
    return S


########################################################################
## Generation of Logical Clifford Operators
########################################################################

def randomCAbin(rng,n,k):
    '''Generate random A and C matrices
    These represent linear combinations of stabilisers to add to a tableau
    to form  a logical Clifford operator with desired action'''
    r = n-k
    xList = [rng.integers(2,size= (2 * r * k + r*(r+1)//2)), randomGLbin(rng,r)]
    return np.hstack(xList)

def randomGLbin(rng,r):
    '''generate a random element of GL_2 on r bits'''
    xList = []
    for i in range(r):
        x = rng.integers(1<<(r-i)-1) + 1
        xList.append(int2bin(x,r-i))
        x = rng.integers(2,size=i)
        xList.append(x)
    return np.hstack(xList)

def bin2CA(x,n,k):
    '''convert binary representation to C and A matrix'''
    r = n-k
    b =  r*(r+1)//2
    A1 = makeSymmetricMatrix(r,x[:b],Sdiag=True)
    a = b
    b = a + r*k
    A2 = np.reshape(x[a:b], (k,r))
    a = b
    b = a + r*k
    C2 = np.reshape(x[a:b], (k,r))
    a = b
    b = a+r*r
    xList = np.reshape(x[a:b],(r,r))
    C1 = bin2GL(xList)
    return ZMatVstack([C1,C2]),ZMatVstack([A1,A2])

def bin2GL(xList):
    '''Convert a binary string to an element of GL_2 on r bits'''
    A = []
    r = len(xList)
    for i in range(r):
        x = xList[i]
        if len(A) == 0:
            A.append(x)
        else:
            H,p = getH(ZMat(A),2,retPivots=True)
            b = ZMatZeros(r)
            p1 = invRange(r,p)
            b[p1] = x[:r-i]
            uA = matMul(x[r-i:],A,2)[0]
            A.append(b ^ uA)
    return ZMat(A)

def sym2UCA(T,k):
    '''convert a symplectic matrix to U,C,A matrix forms
    k is the number of logical qubits'''
    n = len(T) // 2
    r = n-k
    ## logical action
    U = ZMatZeros((2*k,2*k))
    U[:k,:k] = T[r:n,r:n]
    U[:k,k:] = T[r:n,n+r:]
    U[k:,:k] = T[n+r:,r:n]
    U[k:,k:] = T[n+r:,n+r:]

    ## C1: Invertible matrix - stab gen change of basis
    C1 = T[:r,:r]
    ## C2: stabilisers added to LX
    C2 = matMul(T[n:n+r,n+r:].T,C1,2)

    ## A2: stabilisers added to LZ
    A2 = matMul(T[n:n+r,r:n].T,C1,2)
    ## A1: symmetric matrix - stabs added to destabs
    A1 = matMul(C1.T,T[n:n+r,:r],2) ^ matMul(C2.T,A2,2)
    C,A = ZMatVstack([C1,C2]), ZMatVstack([A1,A2])

    print('Sym2UCA check',np.sum(UCA2sym(U,C,A) ^ T)==0)
    return U, C, A

def UCA2sym(U,C,A):
    '''Convert U,C,A matrices to symplectic matrix'''
    n = len(C)
    k = len(U)//2
    r = n-k
    ## Calculate symplectic matrix IxU
    IxU = ZMatI(2*n)
    IxU[r:n,r:n] = U[:k,:k]
    IxU[r:n,n+r:] = U[:k,k:]
    IxU[n+r:,r:n] = U[k:,:k]
    IxU[n+r:,n+r:] = U[k:,k:]

    ## Calculate symplectic matrix UA
    UA = ZMatI(2*n)
    UA[n:,:r] = A
    # A1 = A[:r]
    # print('A1 symmetric',0==np.sum(A1 ^ A1.T))
    A2 = A[r:].T
    UA[n:n+r,r:n] = A2

    ## Calculate symplectic matrix UC
    UC = ZMatI(2*n)
    UC[:n,:r] = C
    In,Cinv = getHU(UC[:n,:n],2)
    # print('C invertible',0 == np.sum(ZMatI(n) ^ In))
    UC[n:,n:] = Cinv.T

    ## output for debugging
    # print('IxU',isSymplectic(IxU))
    # print(ZMatPrint(IxU,tB=2))
    # print('UC',isSymplectic(UC))
    # print(ZMatPrint(UC,tB=2))
    # print('UA',isSymplectic(UA))
    # print(ZMatPrint(UA,tB=2))

    return matMul(IxU,matMul(UC,UA,2),2)


##############################################################################
## Transvection decomposition of Symplectic Matrix
##############################################################################


def transDecomp(U):
    '''Decomposition of symplectic matrix U into 2-transvections, SWAP and single-qubit Clifford layers'''
    ## we will reduce UC to single-qubit Clifford layer
    UC = U.copy()
    n = len(UC)//2
    ## list of 2-transvections
    vList = []
    ## cumulative permutation for SWAP layer
    ixC = np.arange(n)
    ixR = ixC
    for i in range(n):
        ## invertible F matrices in column i
        invList = [j for j in range(i,n) if Fdet(Fmat(UC,j,i)) > 0]
        if len(invList) > 0:
            ## a is smallest j such that Fji is invertible
            a = invList.pop(0)
            ## check if we need to swap rows
            if a > i:
                ## update UC by swapping rows a and i
                ix = np.arange(n)
                ix[[a,i]] = ix[[i,a]]
                UC = matMul(SymSWAP(ix),UC,2)
                ## update ixC and ixR
                ixC[[a,i]] = ixC[[i,a]]
                ixR = ixRev(ixC)
            ## ensure that Fii is the only invertible matrix in col i by pairing invertible matrices in row j,k
            for r in range(len(invList)//2):
                j = invList[2*r]
                k = invList[1+2*r]
                Fji = Fmat(UC,j,i)
                Fki = Fmat(UC,k,i)
                ## calculate a,b,c,d for transvection
                d,c = Fki[:,0]
                I2,FjiInv = getHU(Fji,2)
                a,b = matMul(ZMat([1,0]) ^ matMul(ZMat2D([c,d]),Fki,2),FjiInv,2)[0]
                v = ZMatZeros(2*n)
                v[[j,k,j+n,k+n]] = [a,c,b,d]
                ## check [a,c,b,d] is a 2-transvection
                # print(f'[a,c,b,d]: {a,c,b,d} check:{a + b >0 and c+d>0}')
                ## update UC
                Tv = transvection(v)
                UC = matMul(Tv,UC,2)
                ## check that Fji and Fki are no longer invertible
                # print(f'invertible mat elim i,j,k={i,j,k}; v={v} check:{Fdet(Fmat(UC,j,i)) == 0 and Fdet(Fmat(UC,k,i)) ==0}')
                ## update transvection list - here we move the cumulative SWAP operator through the transvection
                v1 = ZMatPermuteCols(ZMat2D(v),ixR,tB=2)[0]
                vList.append(v1)
        ## eliminate rank 1 F matrices in column i
        for j in range(i+1,n):
            Fji = Fmat(UC,j,i)
            if np.sum(Fji) > 0:
                ## Fii inverse - this may change during elimination process
                I2,FiiInv = getHU(Fmat(UC,i,i),2)
                ## calculate a,b,c,d for transvection
                dcab = matMul(Fji, FiiInv,2)
                a = 0 if sum(dcab[:,0]) == 0 else 1
                b = 0 if sum(dcab[:,1]) == 0 else 1
                d = 0 if sum(dcab[0]) == 0 else 1
                c = 0 if sum(dcab[1]) == 0 else 1
                v = ZMatZeros(2*n)
                v[[i,j,i+n,j+n]] = [a,c,b,d]
                ## update UC
                Tv = transvection(v)
                UC = matMul(Tv,UC,2)
                ## Check that Fji is now zero
                # print(f'rank 1 mat elim i,j={i,j}; v={v} check:{np.sum(Fmat(UC,j,i)) == 0}')
                ## update transvection list - here we move the cumulative SWAP operator through the transvection
                v1 = ZMatPermuteCols(ZMat2D(v),ixR,tB=2)[0]
                vList.append(v1)
    ## Check UC is a product of single-qubit Cliffords
    # D = ZMatZeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         D[i,j] = Fdet(Fmat(UC,i,j))
    # print('UC Check:',np.sum(D ^ ZMatI(n))==0)
    return vList,ixC,UC

def trans2sym(vList,ixC,UC):
    '''convert list of 2-transvections, a qubit permutation and single-qubit Clifford matrix to symplectic matrix'''
    n = len(ixC)
    US = SymSWAP(ixC)
    ## UT is the product of the 2-transvections
    UT = ZMatI(2*n)
    for v in vList:
        UT = matMul(UT,transvection(v),2)
    return matMul(UT,matMul(US,UC,2),2)

def transScore(vList,T):
    '''score transvection by looking at how many gates operator on each qubit'''
    n2 = len(T)
    vList = mod1(ZMatBlockSum(ZMat(vList,n2),tB=2))
    vSum = np.sum(vList,axis=0)
    return tuple(sorted(vSum,reverse=True))
    return (len(vList),tuple(sorted(vSum,reverse=True)))
    B = [rowOverlap(vList,v) for v in vList]
    return tuple(sorted(vSum,reverse=True)) + tuple(sorted(B,reverse=True))


def Fmat(U,i,j):
    '''Return F-matrix: U_{i,j} & U_{i,j+n}\\U_{i+n,j} & U_{i+n,j+n}'''
    n = len(U)//2
    F = ZMatZeros((2,2))
    for r in range(2):
        for c in range(2):
            F[r,c] = U[i + n*r, j + n*c]
    return F

def Fdet(F):
    '''determinant of 2x2 binary matrix'''
    return (F[0,0] * F[1,1]) ^ (F[0,1] * F[1,0])

def SymSWAP(ix):
    '''Symplectic matrix corresponding to qubit permutation ix'''
    n = len(ix)
    M = permMat(ix)
    SOp = ZMatZeros((2*n,2*n))
    SOp[:n,:n] = M
    SOp[n:,n:] = M
    return SOp

######################################################################
## Decomposition of Symplectic Matrix into T = UA @ UB @ UC @ UH
## UA: CXX and sqrt{X} operators
## UB: CZ and S operators
## UC: CNOT operators
## UH: Hadamard operators
######################################################################

def sym2ABCH(T):
    '''Decomposition of Symplectic Matrix into T = UA @ UB @ UC @ UH'''
    n = len(T)//2
    CB = T[:n]
    CB,pX = getH(CB,2,nC=n,retPivots=True)
    B2 = CB[len(pX):,n:]
    B2,h = getH(B2,2,retPivots=True)
    h = ZMat(h)
    TH = XZhad(T,h) if len(h) > 0 else T
    C = TH[:n,:n]
    In, Cinv = getHU(C,2) 
    B = matMul(TH[:n,n:],C.T,2)
    A = matMul(TH[n:,:n],Cinv,2)
    print('T test:', 0==np.sum(T ^ ABCH2sym(A,B,C,h)))
    return A,B,C,h

def ABCH2sym(A,B,C,h):
    '''convert A,B,C,h into a symplectic matrix'''
    n = len(A)
    UA = ZMatI(2*n)
    UA[n:,:n] = A
    UB = ZMatI(2*n)
    UB[:n:,n:] = B
    In,Cinv = getHU(C,2) 
    UC = ZMatZeros((2*n,2*n))
    UC[:n,:n] = C 
    UC[n:,n:] = Cinv.T
    TTest = matMul(matMul(UA,UB,2),UC,2)
    TTest = XZhad(TTest,h)
    return TTest

#####################################################
## Convert stabiliser codes into tableau format
#####################################################

def Stab2Tableau(S0):
    '''Return n,k tableau plus phases for stabilisers in binary form S0'''
    n = len(S0.T) // 2
    ## RREF mod 2 - only consider first n columns, return pivots
    H, Li = getH(S0,2,nC=n,retPivots=True)
    ## independent X checks
    r = len(Li)
    ## reorder rows so pivots are to LHS
    ix = ZMat(Li + invRange(n,Li))
    H = ZMatPermuteCols(H,ix,tB=2)
    ## Swap cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## RREF again
    H,Li = getH(H,2,nC=n,retPivots=True)
    ## number of independent Z checks
    s = len(Li) - r
    ## number of encoded qubits
    k = n - r - s
    ## reorder columns
    ix2 = Li + invRange(n,Li)
    ix = ix[ix2]
    H = ZMatPermuteCols(H,ix2,tB=2)
    ## swap back cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## Extract key matrices
    A2 = H[:r,r+s:n]
    C = H[:r,-k:]
    E = H[r:,-k:]
    ## Form LX/LZ
    LX = ZMatHstack([ZMatZeros((k,r)),E.T,ZMatI(k),C.T, ZMatZeros((k,s+k))])
    LZ = ZMatHstack([ZMatZeros((k,n)),A2.T,ZMatZeros((k,s)),ZMatI(k)])
    ## Form destabilisers
    Rx = ZMatHstack([ZMatZeros((r,n)),ZMatI(r),ZMatZeros((r,n-r))])
    Rz = ZMatHstack([ZMatZeros((s,r)),ZMatI(s),ZMatZeros((s,n+k))])
    ## return qubits to original order
    ixR = ixRev(ix)
    T = ZMatVstack([H,LX,Rx,Rz,LZ])
    T = ZMatPermuteCols(T,ixR,tB=2)
    ## adjust tableau phases to match phases of original stabilisers
    pT = PauliDefaultPhases(T)
    pS0 = PauliDefaultPhases(S0)
    r0, U = HowResU(S0,H,2)
    for i in range(len(S0)):
        p, xz = PauliProd(S0,pS0,U[i])
        pT[i] = p
    return n,k,pT,T


def PauliProd(S,pList,u):
    '''Calculate sign and X/Z components of product of Pauli operators S and phases pList specified by binary vector u'''
    r,n2 = S.shape
    n = n2//2
    p,xz = 0, ZMatZeros(2*n)
    for i in bin2Set(u):
        p = p + pList[i] + 2 * (np.sum(xz[n:] * S[i][:n]) % 2)
        xz ^= S[i]
    return p % 4, xz

def PauliDefaultPhases(S):
    '''Calculate a phase correction which ensures Pauli operators have non-trivial +1 eigenspace'''
    r,n2 = S.shape
    n = n2//2
    return np.sum(S[:,:n] * S[:,n:], axis=-1) % 4

def PauliComm(a,A):
    '''Calculate commutator of Pauli operator a with list of Paulis A'''
    n = len(a)//2
    On = OmegaMat(n)
    p = matMul(ZMat2D(a),matMul(On,A.T,2),2)
    return p[0]