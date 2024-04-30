# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:14:12 2024

@author: gpalo
"""

import numpy as np
import random
import matplotlib.pyplot as plt




def calculateStatistics(A, numPowers):
    currMatrix = A
    frobNorms = []
    singularValuesArr = np.zeros((len(A), numPowers))
    eigValMagnitudesArr = np.zeros((len(A), numPowers))
    
    
    for i in range(numPowers):
        
        
        frobNorm = np.linalg.norm(currMatrix)
        frobNorms.append(frobNorm)
        
        
        currT_curr = currMatrix.T @ currMatrix
        currT_currEigVals = np.linalg.eig(currT_curr)[0]
        currT_currEigVals = np.real(currT_currEigVals)
        currT_currEigVals[currT_currEigVals < 0] = 0.0
        
        
            
            
        singularValues = np.sqrt(currT_currEigVals)
        singularValuesSorted = np.sort(singularValues)
        singularValuesSorted = singularValuesSorted[::-1]
        singularValuesArr[:,i] = singularValuesSorted
        
        eigVals = np.linalg.eig(currMatrix)[0]
        eigValMagnitudes = np.abs(eigVals)
        eigValMagnitudesSorted = np.sort(eigValMagnitudes)
        eigValMagnitudesSorted = eigValMagnitudesSorted[::-1]
        eigValMagnitudesArr[:,i] = eigValMagnitudesSorted
        
        
        currMatrix = np.matmul(currMatrix,A)
    
        
    return frobNorms, singularValuesArr, eigValMagnitudesArr
        
        
        








def doPlots(frobNorms, singularValues, eigValMagnitudes, numPowers, matrixName):
    
    
    

    powersRange = range(1,numPowers+1)

    plt.figure()
    for i in range(len(frobNorms)):    
        plt.plot(powersRange, frobNorms[i,:])
    
    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("FrobNorm of "+matrixName+" A Powers")
    # plt.legend(loc="upper right")
    
    
    
    plt.figure()
    for i in range(len(singularValues)):
        plt.plot(powersRange, singularValues[i,:], label="singularVal"+str(i+1))

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("SingularValues of "+matrixName+" A Powers")
    plt.legend(loc="upper right")
    
    plt.figure()
    plt.plot(powersRange, np.sum(singularValues, axis=0), label="singularValsSummed")

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("SingularValueSums of "+matrixName+" A Powers")
    plt.legend(loc="upper right")
    
    

    plt.figure()
    plt.plot(powersRange, np.prod(singularValues, axis=0), label="singularValsProduct")

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("SingularValueProducts of "+matrixName+" A Powers")
    plt.legend(loc="upper right")
    
    ################## do eigenvalue magnitudes
    
    plt.figure()
    for i in range(len(eigValMagnitudes)):
        plt.plot(powersRange, eigValMagnitudes[i,:], label="eigValMagnitude"+str(i+1))

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("EigValMagnitudes of "+matrixName+" A Powers")
    plt.legend(loc="upper right")
    
    plt.figure()
    plt.plot(powersRange, np.sum(eigValMagnitudes, axis=0), label="singularValsSummed")

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("EigValMagnitudeSums of "+matrixName+" A Powers")
    plt.legend(loc="upper right")
    


    plt.figure()
    plt.plot(powersRange, np.prod(eigValMagnitudes, axis=0), label="singularValsProduct")

    plt.xlabel("Power")
    plt.ylabel("Value")
    plt.title("EigValMagnitudeProducts of "+matrixName+" A Powers")
    plt.legend(loc="upper right")


def generateAdense(MATRIX_SIZE):
    
    Adense = np.random.uniform(-1.0,1.0,(MATRIX_SIZE,MATRIX_SIZE))
        
    eigVals = np.linalg.eig(Adense)[0]
    eigValsAbs = np.abs(eigVals)

    maxEigValAbs = np.max(eigValsAbs)

    Adense = .95*Adense/maxEigValAbs

    eigValsCheck = np.linalg.eig(Adense)[0]
    
    return Adense, "dense"


def generateAUpperTriangular(MATRIX_SIZE):
    
    Adense = np.random.uniform(-1.0,1.0,(MATRIX_SIZE,MATRIX_SIZE))
    AupperTriangular = np.triu(Adense)
    eigVals = np.linalg.eig(AupperTriangular)[0]
    eigValsAbs = np.abs(eigVals)

    maxEigValAbs = np.max(eigValsAbs)

    AupperTriangular = .95*AupperTriangular/maxEigValAbs

    eigValsCheck = np.linalg.eig(AupperTriangular)[0]
    
    return AupperTriangular, "upperTriangular"

    

def doAnalysis(AgeneratingFunc, MATRIX_SIZE,\
               NUM_RESULTS_AVERAGING, NUM_POWERS,
               DO_SUBTRACTION):
    
    frobNormsArr = np.zeros((NUM_RESULTS_AVERAGING, NUM_POWERS))

    singularValsArr = np.zeros((NUM_RESULTS_AVERAGING, MATRIX_SIZE, NUM_POWERS))
    
    eigValMagnitudesArr = np.zeros((NUM_RESULTS_AVERAGING, MATRIX_SIZE, NUM_POWERS))
    # singularValSumsArr = np.zeros((NUM_RESULTS_AVERAGING, NUM_POWERS))
    
    
    
    for i in range(NUM_RESULTS_AVERAGING):
    
        A, matrixName = AgeneratingFunc(MATRIX_SIZE)

        if DO_SUBTRACTION:
            Kp = np.zeros(A.shape)
            C = np.zeros(A.shape)
            
            A = A-(Kp @ C)
            
            
        # print("eigVals of normalized A:", eigValsCheck)

        AFrobNorms, ASingularValues, AEigValMagnitudes = calculateStatistics(A,NUM_POWERS)
    
        frobNormsArr[i] = AFrobNorms
        singularValsArr[i] = ASingularValues
        eigValMagnitudesArr[i] = AEigValMagnitudes
    
    # frobNormsAveraged = np.average(frobNormsArr, axis=0)
    singularValsAveraged = np.average(singularValsArr, axis=0)
    eigValMagnitudesAveraged = np.average(eigValMagnitudesArr, axis=0)

    doPlots(frobNormsArr, singularValsAveraged, eigValMagnitudesAveraged, NUM_POWERS, matrixName)

    

if __name__ == "__main__":
    
    random.seed(10)   
    np.random.seed(10)
    
    MATRIX_SIZE=10
    NUM_RESULTS_AVERAGING=1
    NUM_POWERS=50
                   
                   
    
    doAnalysis(generateAdense, MATRIX_SIZE,\
                    NUM_RESULTS_AVERAGING, NUM_POWERS)

        
    # doAnalysis(generateAUpperTriangular, MATRIX_SIZE,\
    #                 NUM_RESULTS_AVERAGING, NUM_POWERS) 
        
        
        
        
        
        
        
        
def solve_ricc(A, W):  # solve the Riccati equation for the steady state solution
    L, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    Pi = (V @ (
            (Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)
    ) @ V.T).real
    return Pi
        
        
        
        
        
        
        