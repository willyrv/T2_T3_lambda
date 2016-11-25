# -*- coding: utf-8 -*-

import os, re
import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.linalg import expm

path2ms = './'

def simulateT2T3(ms_command):
    """
    Simulate values of (T2, T3) using ms
    """
    (T2_values, T3_values) = ([], [])
    ms_results = os.popen(os.path.join(path2ms, ms_command)).read()
    # We get only the tree height and the tree length
    pattern = 'time:\t[0-9]*\.[0-9]*\t[0-9]*\.[0-9]*'
    p = re.compile(pattern)
    for line in p.findall(ms_results):
        values = line.split('\t')
        (tree_height, tree_length) = (float(values[1]), float(values[2]))
        T2_values.append(3*tree_height - tree_length)
        T3_values.append(tree_length - 2*tree_height)
    return (T2_values, T3_values)
    
def construct_cdf_pdf(data, bins_vector):
    """
    Compute an approximation of the cumulative distribution (cdf) and
    the probability density function (pdf) based on the data and a
    predefinded vector (bins_vector)
    """
    pdf = np.histogram(data, bins=bins_vector, density=True)[0]
    v = np.array(bins_vector)
    differences = v[1:]-v[:-1]
    temp_cdf = pdf * differences
    cdf = temp_cdf.cumsum()
    return (pdf, cdf)
    
def evaluate_theor_T2(u, T3pdf, T3cdf, bins_vector):
    # The step in bins_vector should be constant
    step = bins_vector[1]-bins_vector[0]
    ui = np.int(np.true_divide(u, step))
    if ui == 0:
        return 0
    else:
        d0 = (np.int(np.true_divide(u, step))+1)*step -u
        d1 = u - (np.int(np.true_divide(u, step)))*step
        sd0 = d0*sum(np.exp(np.true_divide(np.log(1-T3cdf[ui:]) - np.log(1-T3cdf[:-ui]),3)) * T3pdf[:-ui])
        sd1 = d1*sum(np.exp(np.true_divide(np.log(1-T3cdf[ui+1:]) - np.log(1-T3cdf[:-ui-1]),3)) * T3pdf[:-ui-1])
        return 1 - (sd0+sd1)

def compute_Q3(n, M):
    """
    Compute the Q-matrix of T3, given n and M
    """
    (n, M) = (float(n), float(M))
    return np.matrix([[-3*(1 + M/2), 3*M/2, 0, 3, 0], 
      [M/(2*n -2), -1 - M*(2*n-3)/(2*n-2), M*(n-2)/(n-1), 0, 1],
     [0, 3*M/(n-1), -3*M/(n-1), 0, 0],
     [0,0,0,0,0], 
      [0,0,0,0,0]])
      
def compute_Q2(n, M):
    """
    Compute the Q-matrix of T2, given n and M
    """
    # Q-matrix for T2
    (n, M) = (float(n), float(M))
    return np.matrix([[-M-1, M, 1],
               [M/(n-1), -M/(n-1), 0], 
               [0,0,0]])
               
def evaluate_at_u_cond_on_t(t, u, Q2, Q3):
    """
    Evaluates the cumulative distribution function of T2 under the structured
    model (Fstr) and the panmictic model (Fchang). This two models satisfy
    that they has the same distribution of T3 (the first coalescent event
    of three genes).
    The cumulative distribution functions are computed conditionning on that
    the first coalescent event of 3 genes has occured at time t.
    """
    # Evaluate the transition semigroup for T3 at t
    P3t = linalg.expm(t*Q3)    
    # Evaluate the transition semigroup of T2 under structure at u
    P2u = linalg.expm(u*Q2)
    Fstr = (3*P2u[0,2]*P3t[0,0] + P2u[1,2]*P3t[0,1])/(3*P3t[0,0] + P3t[0,1])
    P3tplusu = linalg.expm((t+u)*Q3)
    Fchang = 1 - ((1-P3tplusu[0,3] - P3tplusu[0,4])/(1-P3t[0,3]-P3t[0,4]))**(1./3)
    return [Fstr, Fchang]
    
def compute_cdfT2_Fstr_Fchang(t, u, n, M):
    # Compute Q-matrices for the n-islands model
    Q3 = compute_Q3(n, M)
    Q2 = compute_Q2(n, M)
    return evaluate_at_u_cond_on_t(t, u, Q2, Q3)
    
def compute_diff_Fstr_Fchang_times_densityT3(t, u, Q2, Q3):
    """
    Compute the differences of both functions at a given u, conditionning
    on that T3=t
    """
    # Evaluate the transition semigroup for T3 at t
    P3t = linalg.expm(t*Q3)    
    
    # Compute the density at t using the transition semigroup
    fT3_t = 3*P3t[0,0] + P3t[0,1]
    
    # Evaluate the transition semigroup of T2 under structure at u
    P2u = linalg.expm(u*Q2)
    Fstr = (3*P2u[0,2]*P3t[0,0] + P2u[1,2]*P3t[0,1])/fT3_t
    P3tplusu = linalg.expm((t+u)*Q3)
    Fchang = 1 - ((1-P3tplusu[0,3] - P3tplusu[0,4])/(1-P3t[0,3]-P3t[0,4]))**(1./3)
    return abs(Fstr-Fchang)*fT3_t
    
def computeP3t_didier(t, n, M):
    (n, M) = (float(n), float(M))
    p = [1, (1./2)*(5*M*n+8*n-8)/(n-1), 
         (3./2)*(M**2*n**2+3*M*n**2+M*n+2*n**2-4*M-4*n+2)/(n-1)**2,
            (9./2)*M*(M*n+2*n-2)/(n-1)**2]
    roots = np.roots(p)
    mat_B = [[0, 0, 0, (M+2*n-2)/(M*n+2*n-2), M*(n-1)/(M*n+2*n-2)], [0, 0, 0, M/(M*n+2*n-2), (M+2)*(n-1)/(M*n+2*n-2)], [0, 0, 0, M/(M*n+2*n-2), (M+2)*(n-1)/(M*n+2*n-2)], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    temp_P3t = np.matrix(mat_B)
    for lambda_value in roots:
        mat_A_lambda = [[2*M*lambda_value*n**2+2*lambda_value**2*n**2+M*lambda_value*n-4*lambda_value**2*n+2*lambda_value*n**2+3*M**2-3*M*lambda_value+6*M*n+2*lambda_value**2-4*lambda_value*n-6*M+2*lambda_value, (3*lambda_value*n+9*M-3*lambda_value)*M*(n-1), (3*n-6)*M**2*(n-1), -(3*M**3*n**2+5*M**2*lambda_value*n**2+2*M*lambda_value**2*n**2+4*M*lambda_value*n**3+4*lambda_value**2*n**3-5*M**2*lambda_value*n-4*M*lambda_value**2*n-12*lambda_value**2*n**2+4*lambda_value*n**3+12*M**2*n+2*M*lambda_value**2-12*M*lambda_value*n+12*M*n**2+12*lambda_value**2*n-12*lambda_value*n**2-12*M**2+8*M*lambda_value-24*M*n-4*lambda_value**2+12*lambda_value*n+12*M-4*lambda_value)/(M*n+2*n-2), -(n-1)*M*(3*M**2*n**2+5*M*lambda_value*n**2+2*lambda_value**2*n**2-5*M*lambda_value*n+6*M*n**2-4*lambda_value**2*n+8*lambda_value*n**2+6*M*n+2*lambda_value**2-16*lambda_value*n-12*M+8*lambda_value)/(M*n+2*n-2)], [M*(lambda_value*n+3*M-lambda_value), (3*M+2*lambda_value+6)*(lambda_value*n+3*M-lambda_value)*(n-1), (n-2)*(3*M+2*lambda_value+6)*M*(n-1), -M*(3*M**2*n**2+5*M*lambda_value*n**2+2*lambda_value**2*n**2-5*M*lambda_value*n+6*M*n**2-4*lambda_value**2*n+8*lambda_value*n**2+6*M*n+2*lambda_value**2-16*lambda_value*n-12*M+8*lambda_value)/(M*n+2*n-2), -(n-1)*(3*M**3*n**2+5*M**2*lambda_value*n**2+2*M*lambda_value**2*n**2-5*M**2*lambda_value*n+12*M**2*n**2-4*M*lambda_value**2*n+16*M*lambda_value*n**2+4*lambda_value**2*n**2+2*M*lambda_value**2-24*M*lambda_value*n+12*M*n**2-8*lambda_value**2*n+12*lambda_value*n**2-12*M**2+8*M*lambda_value+4*lambda_value**2-24*lambda_value*n-12*M+12*lambda_value)/(M*n+2*n-2)], [3*M**2, (9*M+6*lambda_value+18)*M*(n-1), (3*M**2*n+5*M*lambda_value*n+2*lambda_value**2*n-6*M**2-6*M*lambda_value+9*M*n-2*lambda_value**2+8*lambda_value*n-12*M-8*lambda_value+6*n-6)*(n-1), -M*(3*M**2*n**2+5*M*lambda_value*n**2+2*lambda_value**2*n**2-5*M*lambda_value*n+9*M*n**2-4*lambda_value**2*n+8*lambda_value*n**2+3*M*n+2*lambda_value**2-16*lambda_value*n+6*n**2-12*M+8*lambda_value-12*n+6)/(M*n+2*n-2), -(n-1)*(3*M**3*n**2+5*M**2*lambda_value*n**2+2*M*lambda_value**2*n**2-5*M**2*lambda_value*n+15*M**2*n**2-4*M*lambda_value**2*n+18*M*lambda_value*n**2+4*lambda_value**2*n**2-3*M**2*n+2*M*lambda_value**2-26*M*lambda_value*n+24*M*n**2-8*lambda_value**2*n+16*lambda_value*n**2-12*M**2+8*M*lambda_value-18*M*n+4*lambda_value**2-32*lambda_value*n+12*n**2-6*M+16*lambda_value-24*n+12)/(M*n+2*n-2)], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        mat_A_lambda = np.matrix(mat_A_lambda)        
        denA=(6*n**2-12*n+6)*lambda_value**2+(10*M*n**2-10*M*n+16*n**2-32*n+16)*lambda_value+3*M**2*n**2+9*M*n**2+3*M*n+6*n**2-12*M-12*n+6
        temp_P3t = temp_P3t + mat_A_lambda * np.exp(t*lambda_value)/denA
    return temp_P3t

def computeP3t_diag(t, Q3):
    """
    Compute the exponential of Q3 by diagonalizing Q3
    """
    # Diagonalize the matrix Q3
    [eig_values, P] = np.linalg.eig(Q3)
    P_inv = np.linalg.inv(P)
    eig_values_exp = np.exp(t*np.array(eig_values))
    D = np.diag(eig_values_exp)
    return(P * D * P_inv)

def function2integrate(t, u, Q3):
    """
    Evaluates the function that will be integrated over all t.
    Assume that t is a one-dimension array
    """
    # Evaluate the transition semigroup for T3 at t
    
    # Using the Didier's method
    #P3t = computeP3t_didier(t, n, M)
    #P3tplusu = computeP3t_didier(t+u, n, M)
    
    # Using the diagonalization of the matrix Q3    
    P3t = computeP3t_diag(t, Q3)
    P3tplusu = computeP3t_diag((t+u), Q3)
    
    #Using the expm method of scipy.linalg
    #P3t = expm(t*Q3)
    #P3tplusu = expm((t+u)*Q3)
    
    factor1 = np.true_divide(1-P3tplusu[0,3]-P3tplusu[0,4], 1-P3t[0,3]-P3t[0,4])
    factor2 = 3*P3t[0, 0] + P3t[0, 1]
    return(factor1**(1./3) * factor2)
    
    
def cdf_T2_3_lambda(u, n, M):
    """
    Evaluate the cdf of T2 after T3 under a panmictic model with a lambda
    function (or IICR) which corresponds to n and M.
    """
    # Calculate the Q3 matrix
    Q3 = compute_Q3(n, M)    
    return 1 - quad(function2integrate, 0, np.infty, args=(u, Q3))[0]

def cdf_T2_3_str(u, n, M):
    """
    Compute the theoretical cdc of T2 after T3 in a n-island model
    """
    
    Q2 = compute_Q2(n, M) 
    P2u = linalg.expm(u*Q2)
    M = float(M)
    n = float(n) 
    return (M + 2*n-2)/ (M*n + 2*n-2)*  P2u[0,2] + (1- (M + 2*n-2)/(M*n + 2*n-2))*P2u[1,2]
    
def cdf_T3_str(t, n, M):
    """
    Compute the cumulative distribution function of T3 (the first coalescence event
    between two genes in a sample of 3 genes) for the n-islands model
    """

    Q3 = compute_Q3(n, M) 
    P3t = linalg.expm(t*Q3)
    return P3t[0,3] + P3t[0,4]

def integrate_all_t(u, n, M):
    """
    Integrate out all possible t values (the values of T3)
    """
    # Compute Q-matrices for the n-islands model
    Q3 = compute_Q3(n, float(M))
    Q2 = compute_Q2(n, float(M))
    return quad(compute_diff_Fstr_Fchang_times_densityT3, 0, 100, args=(u, Q2, Q3))
