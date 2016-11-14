"""
Prepare Sparse Matrix for Sparse Affinity Propagation Clustering (SAP)

Script from pysapc.
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from pysapc import sparseAP_cy # cython for calculation


def copySym(rows, cols, rowBased_data, singleRowInds):
    """
    For single col items or single row items, copy sym minimal value
    For example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...] is s(A,B),
    then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...] (except s(B,A), because if we copy s(B,A), for 'A' we still only have one data point)
    """
    copy_row_array,copy_col_array,copy_data_array=sparseAP_cy.copySingleRows(rows,cols,rowBased_data,singleRowInds)
    df = pd.DataFrame(zip(copy_row_array,copy_col_array,copy_data_array), columns=['row', 'col', 'data'])
    copy_row_list,copy_col_list,copy_data_list=[],[],[]
    for ind in singleRowInds:
        copyData=df[(df.col==ind) & (df.row!=ind)].sort(['data']).copy()
        copyData_min=copyData[0:1]
        copy_row_list+=list(copyData_min.col)
        copy_col_list+=list(copyData_min.row)
        copy_data_list+=list(copyData_min.data)
    rows=np.concatenate((rows,copy_row_list))
    cols=np.concatenate((cols,copy_col_list))
    rowBased_data=np.concatenate((rowBased_data,copy_data_list))
    return rows,cols,rowBased_data

def rmSingleSamples(rows, cols, rowBased_data, n_samples):
    """
    Affinity/similarity matrix does not need be symmetric, that is s(A,B) does not need be same as s(B,A).
    Also since Affinity/similarity matrix is sparse, it could be that s(A,B) exist but s(B,A) does not exist in the sparse matrix.
    For the FSAPC to work, specifically in computation of R and A matrix, each row/column of Affinity/similarity matrix should have at least two datapoints.
    So in FSAPC, we first remove samples that do not have affinity/similarity with other samples, that is samples that only have affinity/similarity with itself
    And we remove samples only have one symmetric datapoint, for example for sample 'B' only s(B,C) exist and for sample 'C' only s(C,B) exist
    In these two cases, these samples are removed from FSAPC computation and their examplers are set to themself.
    For samples that only have one data (affinity/similarity) with others, For example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...] is s(A,B),
    and there exist at least one value in [s(A,A),s(C,A),s(D,A)...] (except s(B,A), because if we copy s(B,A), for 'A' we still only have one data point)
    then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...]
    n_samples is the number of samples of orignail input data
    """
    # find rows and cols that only have one datapoint
    # singleRowInds=set(sparseAP_cy.singleItems(rows))
    # singleColInds=set(sparseAP_cy.singleItems(cols))

    unique, counts = np.unique(rows, return_counts=True)
    singleRowInds = set(unique[counts < 2])
    unique, counts = np.unique(cols, return_counts=True)
    singleColInds = set(unique[counts < 2])

    # assert sorted(singleRowInds) == sorted(rows_one_element)
    # in case every col/row have more than one datapoint, return original data
    if len(singleRowInds) == 0 and len(singleColInds) == 0:
        return rows, cols, rowBased_data, \
            None, None, n_samples

    # samples that have one datapoint in row and col are samples only have
    # affinity/similarity with itself
    singleSampleInds = singleRowInds & singleColInds

    # remove samples that only have affinity/similarity with itself
    # or only have one symmetric datapoint, for example for sample 'B' only
    # s(B,C) exist and for sample 'C' only s(C,B) exist
    # in these two cases, these samples are removed from FSAPC computation and
    # their examplers are set to themself.
    if len(singleSampleInds) > 0:
        # row indexs that left after remove single samples
        # rowLeft_old = sorted(list(set(range(n_samples)) - singleSampleInds))
        mask = np.ones(n_samples, dtype=bool)
        mask[np.array(singleSampleInds)] = False
        rowLeft = np.arange(n_samples)[mask]
        # assert rowLeft == rowLeft_old
        # map of original row index to current row index(after remove rows/cols
        # that only have single item)
        rowOriLeftDict, rowLeftOriDict = {}, {}
        for left, ori in enumerate(rowLeft):
            rowOriLeftDict[ori] = left
            rowLeftOriDict[left] = ori

        # remove single elements outside the diagonal
        mask = np.ones(rows.shape[0], dtype=bool)
        mask[singleSampleInds] = False
        idx_to_remove = np.logical_or(rows == cols, mask)
        rows = rows[idx_to_remove]
        cols = cols[idx_to_remove]
        rowBased_data = rowBased_data[idx_to_remove]

        # rows, cols, rowBased_data = sparseAP_cy.removeSingleSamples(
        #     rows, cols, rowBased_data, singleSampleInds)
        # assert set(rows) == set(rows)
        # assert set(cols) == set(cols)
        # assert set(data) == set(rowBased_data)
    else:  # no samples are removed
        rowLeftOriDict = None

    # for samples that need copy a minimal value to have at least two
    # datapoints in row/column
    # for samples that row have single data point, copy minimal value of this
    # sample's column
    singleRowInds = singleRowInds - singleSampleInds
    if len(singleRowInds) > 0:
        rows, cols, rowBased_data = copySym(
            rows, cols, rowBased_data, singleRowInds)
    # for samples that col have single data point, copy minimal value of this sample's row
    singleColInds = singleColInds - singleSampleInds
    if len(singleColInds) > 0:
        cols, rows, rowBased_data = copySym(
            cols, rows, rowBased_data, singleColInds)

    # change row, col index if there is any sample removed
    if len(singleSampleInds) > 0:
        changeIndV = np.vectorize(lambda x: rowOriLeftDict[x])
        rows = changeIndV(rows)
        cols = changeIndV(cols)

    # rearrange based on new row index and new col index
    sortedLeftOriInd = np.lexsort((cols, rows)).astype(np.int)

    rows = sparseAP_cy.npArrRearrange_int_para(
        rows, sortedLeftOriInd)
    cols = sparseAP_cy.npArrRearrange_int_para(
        cols, sortedLeftOriInd)
    rowBased_data = sparseAP_cy.npArrRearrange_float_para(
        rowBased_data, sortedLeftOriInd)
    # rows2 = rows[sortedLeftOriInd]
    # cols2 = cols[sortedLeftOriInd]
    # rowBased_data2 = rowBased_data[sortedLeftOriInd]
    # assert rowBased_data1 == rowBased_data2
    # assert cols1 == cols2
    # assert rows1 == rows2

    return rows, cols, rowBased_data, rowLeftOriDict, \
        singleSampleInds, n_samples - len(singleSampleInds)
