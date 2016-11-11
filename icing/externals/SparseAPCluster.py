"""
Sparse Affinity Propagation (SAP)
Designed for large data set using scipy sparse matrix(affinity/similarity matrix)
Speed optimized with cython
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np
from datetime import datetime
import sparseAP_cy # cython for calculation speed optimization
import sparseMatrixPrepare

#########################################################################



def updateR_cython(S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping):
    """
    Update Responsibilities Matrix (R)
    """
    as_data=A_rowbased_data_array+S_rowBased_data_array
    as_max_data_arr=sparseAP_cy.updateR_maxRow(as_data,row_indptr)
    r_new_data_arr=S_rowBased_data_array-as_max_data_arr
    r_row_data=(r_new_data_arr*(1.0-damping)) + (R_rowbased_data_array*damping)
    return r_row_data

def updateA_cython(A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping):
    """
    Update Availabilities Matrix (A)
    """
    A_colbased_data_array=sparseAP_cy.npArrRearrange_float(A_rowbased_data_array,row_to_col_ind_arr)
    R_colbased_data_array=sparseAP_cy.npArrRearrange_float(R_rowbased_data_array,row_to_col_ind_arr)
    r_col_data=np.copy(R_colbased_data_array)
    r_col_data[r_col_data<0]=0
    r_col_data[kk_col_index]=R_colbased_data_array[kk_col_index]
    a_col_data_new=sparseAP_cy.updateA_col(r_col_data,col_indptr,kk_col_index)
    a_col_data=(a_col_data_new*(1.0-damping)) + (A_colbased_data_array*damping)
    A_rowbased_data_array=sparseAP_cy.npArrRearrange_float(a_col_data,col_to_row_ind_arr)
    return A_rowbased_data_array

def updateR_cython_para(S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping):
    """
    Update Responsibilities Matrix (R), with cython multiprocessing.
    """
    as_data=A_rowbased_data_array+S_rowBased_data_array
    as_max_data_arr=sparseAP_cy.updateR_maxRow_para(as_data,row_indptr)
    r_new_data_arr=S_rowBased_data_array-as_max_data_arr
    r_row_data=(r_new_data_arr*(1.0-damping)) + (R_rowbased_data_array*damping)
    return r_row_data


def updateA_cython_para(A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping):
    """
    Update Availabilities Matrix (A), with cython multiprocessing.
    """
    A_colbased_data_array=sparseAP_cy.npArrRearrange_float_para(A_rowbased_data_array,row_to_col_ind_arr)
    R_colbased_data_array=sparseAP_cy.npArrRearrange_float_para(R_rowbased_data_array,row_to_col_ind_arr)
    r_col_data=np.copy(R_colbased_data_array)
    r_col_data[r_col_data<0]=0
    r_col_data[kk_col_index]=R_colbased_data_array[kk_col_index]
    a_col_data_new=sparseAP_cy.updateA_col_para(r_col_data,col_indptr,kk_col_index)
    a_col_data=(a_col_data_new*(1.0-damping)) + (A_colbased_data_array*damping)
    A_rowbased_data_array=sparseAP_cy.npArrRearrange_float_para(a_col_data,col_to_row_ind_arr)
    return A_rowbased_data_array


def denseToSparseAbvCutoff(self, denseMatrix, cutoff):
    """
    Remove datas in denseMatrix that is below cutoff,
    Convert the remaining datas into sparse matrix.
    Parameters:
    ----------------------
    denseMatrix: dense numpy matrix

    cutoff: int or float

    Returns
    ----------------------
    Scipy csr_matrix

    """
    maskArray=denseMatrix>=cutoff
    sparseMatrix=csr_matrix( (np.asarray(denseMatrix[maskArray]).reshape(-1),np.nonzero(maskArray)),\
                shape=denseMatrix.shape)
    return sparseMatrix


def denseToSparseTopPercentage(self, denseMatrix, percentage=10.0):
    """
    Keep top percentage (such as 10%) of data points,
    remove all others. Convert into sparse matrix.
    Parameters:
    ----------------------
    denseMatrix: dense numpy matrix

    percentage: float, default is 10.0
        percentage of top data points to keep. default is 10.0% that is for 10000 data points keep top 1000.

    Returns
    ----------------------
    Scipy csr_matrix

    """
    rowN,colN=denseMatrix.shape
    totalN=rowN*colN
    topN=min(int(totalN*(percentage/100.0)), totalN)
    arr=np.array(denseMatrix.flatten())[0]
    cutoff=arr[arr.argsort()[-(topN)]]
    sparseMatrix=self.denseToSparseAbvCutoff(denseMatrix,cutoff)
    return sparseMatrix
