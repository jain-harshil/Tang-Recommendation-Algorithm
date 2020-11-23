import numpy as np
from numpy import linalg as la
import time
import os


def probab_ls(m, n, A):
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    A_Frobenius = np.sqrt(np.sum(row_norms))

    rows_prob = np.zeros(m)

    for i in range(m):
        rows_prob[i] = row_norms[i] / A_Frobenius**2

    col_prob = np.zeros((m, n))
    for i in range(m):
        col_prob[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

    return row_norms, rows_prob, col_prob, A_Frobenius

def sam_C(A, m, n, r, c, row_norms, rows_prob, col_prob, A_Frobenius):
	# to build matrix C by performing sampling rows and columns A

    tic = time.time()
    rows = np.random.choice(m, r, replace=True, p=rows_prob)

    columns = np.zeros(c, dtype=int)
    for j in range(c):
        i = np.random.choice(rows, replace=True)
        columns[j] = np.random.choice(n, 1, p=col_prob[i])

    toc = time.time()
    rt_sampling_C = toc - tic

    R_row = np.zeros(n)
    LS_prob_columns_R = np.zeros((r, n))

    for s in range(r):
        R_row[:] = A[rows[s], :] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))
        R_row_norm = np.abs(la.norm(R_row[:]))**2
        LS_prob_columns_R[s, :] = [np.abs(k)**2 / R_row_norm for k in R_row[:]]

    tic = time.time()

    R_C = np.zeros((r, c))
    C = np.zeros((r, c))

    for s in range(r):
        for t in range(c):
            R_C[s, t] = A[rows[s], columns[t]]

        R_C[s,:] = R_C[s,:] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))

    column_norms = np.zeros(c)

    for t in range(c):
        for s in range(r):
            column_norms[t] += np.abs(R_C[s, t])**2

    for t in range(c):
        C[:, t] = R_C[:, t] * (A_Frobenius / np.sqrt(column_norms[t])) / np.sqrt(c)

    toc = time.time()
    rt_building_C = toc - tic

    tic = time.time()
  
    w, sigma, vh = la.svd(C, full_matrices=False)

    toc = time.time()
    rt_svd_C = toc - tic

    return w, rows, sigma, vh, LS_prob_columns_R, rt_sampling_C, rt_building_C, rt_svd_C

def samp_reccsys(A, user, n, samples, rank, r, w, rows, sigma, row_norms, col_prob, A_Frobenius):
    reps = 10

    coefficients = np.zeros((reps, rank))

    for i in range(reps):   
        for l in range(rank): 
            X = np.zeros(samples)  
            for k in range(samples): 
                sample_j = np.random.choice(n, 1, p=col_prob[user])[0]
                v_j = 0
                for s in range(r):
                    v_j += A[rows[s], sample_j] * w[s, l] / (np.sqrt(row_norms[rows[s]]))
                v_j = v_j * A_Frobenius / (np.sqrt(r) * sigma[l])
                X[k] = (row_norms[user]*v_j) / (A[user, sample_j])
            coefficients[i, l] = np.mean(X)

    lambdas = np.zeros(rank)
    for l in range(rank):
        lambdas[l] = np.median(coefficients[:, l])

    return lambdas

def rej_sample(A, r, n, rows, row_norms, LS_prob_columns_R, A_Frobenius, w_vector, w_norm):
	# Rejection Sampling
    keep_going = True
    out_j = 0
    counter = 0
    while keep_going:
        counter += 1
        i_sample = np.random.choice(r)
        j_sample = np.random.choice(n, 1, p=LS_prob_columns_R[i_sample])[0]
        R_j = np.zeros(r)
        for s in range(r):
            R_j[s] = A[rows[s], j_sample] / np.sqrt(row_norms[rows[s]])
        R_j = (A_Frobenius/np.sqrt(r)) * R_j
        R_j_norm = la.norm(R_j)
 
        Rw_dot = np.dot(R_j, w_vector)

        prob = (Rw_dot / (w_norm * R_j_norm))**2

        coin = np.random.binomial(1, prob)
        if coin == 1:
            out_j = j_sample
            keep_going = False

    return int(out_j), counter

def app_sol(A, rank, r, w, rows, sigma, row_norms, A_Frobenius, lambdas, comp):
    approx_value = 0
    for l in range(rank):
        v_comp = 0
        for s in range(r):
            v_comp += A[rows[s], comp] * w[s, l] / np.sqrt( row_norms[ rows[s] ] )
        v_comp = v_comp * A_Frobenius / (np.sqrt(r) * sigma[l])
        approx_value += v_comp * lambdas[l]

    return approx_value
           
def recomm_syst(A, user, r, c, rank, Nsamples, NcompX):
    m_rows, n_cols = np.shape(A)

    tic = time.time()

    LS = probab_ls(m_rows, n_cols, A)

    toc = time.time()

    rt_ls_prob = toc - tic

    svd_C = sam_C(A, m_rows, n_cols, r, c, *LS[0:4])
    w = svd_C[0]
    sigma = svd_C[2]

    ul_approx = np.zeros((m_rows, rank))
    vl_approx = np.zeros((n_cols, rank))
    for l in range(rank):

    	m, n = A.shape
    	u_approx = np.zeros(m)
    	v_approx = np.zeros(n)
    	factor = LS[3] / ( np.sqrt(r) * sigma[l] )
    	for s in range(r):
        	v_approx[:] += ( A[svd_C[1][s], :] / np.sqrt(LS[0][svd_C[1][s]]) ) * w[s, l]
    	v_approx[:] = v_approx[:] * factor

    	u_approx = (A @ v_approx) / sigma[l]

    	ul_approx[:, l] = u_approx
    	vl_approx[:, l] = v_approx

    tic = time.time()

    lambdas = samp_reccsys(A, user, n_cols, Nsamples, rank, r, *svd_C[0:3], LS[0], *LS[2:4] )

    toc = time.time()
    rt_sampling_me = toc - tic
    tic = time.time()

    w_vector = np.zeros(r)
    for l in range(rank):
        w_vector[:] += (lambdas[l] / sigma[l]) * w[:, l]

    w_norm = la.norm(w_vector)

    sampled_comp = np.zeros(NcompX, dtype=np.uint32)
    n_of_rejected_samples = np.zeros(NcompX, dtype=np.uint32)
    x_tilde = np.zeros(NcompX)

    for t in range(NcompX):
        sampled_comp[t], n_of_rejected_samples[t] = \
            rej_sample(A, r, n_cols, svd_C[1], LS[0], svd_C[4], LS[3], w_vector, w_norm)

    toc = time.time()
    rt_sampling_sol = toc - tic

    for t in range(NcompX):
        x_tilde[t] = app_sol(A, rank, r, w, svd_C[1], svd_C[2],
                                          LS[0], LS[3], lambdas, sampled_comp[t])

    FKV = [r, c, rank, sigma, ul_approx, vl_approx]

    # print_output(*FKV, *MC, *RS, *RT)
    return sampled_comp, x_tilde

A = np.load('movies.npy')  # 611 users 9724 movies

user = 517

rank = 10 # low rank approx - SVD
r = 420 # sampled rows
c = 4200 # sampled columns
Nsamples = 10 # samples to estimate coefficents 
NcompX = 10 # number of entries to be sampled.
sampled_comp, x = recomm_syst(A, user, r, c, rank, Nsamples, NcompX)

print(sampled_comp)
