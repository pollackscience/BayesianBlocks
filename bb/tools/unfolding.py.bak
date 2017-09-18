#! usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg


def calc_granularity(hist2d_output):
    '''Calulates the granularity of a plt.hist2d object.

    The granularity approaches 0 when the histogram approaches a single bin, and 1 when the
    geometric area of the diagonal is negligble with respect to the range of the histogram.

    Assumes the hist2d is square.
    '''
    bex = hist2d_output[1]
    length = max(bex) - min(bex)
    area = length**2
    diag_area = 0
    for i in range(len(bex)-1):
        diag_area += (bex[i+1]-bex[i])**2

    return (1-diag_area/area)


def calc_diagonality(hist2d_output):
    '''Calulates the diagonality of a plt.hist2d object.

    The diagonality approaches 0 when the diagonal elements contain none of the total histogram
    content, and 1 when diagonal elements contain the total histogram content.

    Assumes the hist2d is square.
    '''
    bc = hist2d_output[0]
    bc_total = bc.sum()
    bc_diag = 0
    for i in range(len(bc)):
        bc_diag += bc[i, i]

    return (bc_diag/bc_total)


def calc_response_matrix(df, bins, lims):
    bc, bex, bey = np.histogram2d(df['reco'], df['gen'], bins, range=(lims, lims))
    for i in range(bc.shape[0]):
        col_sum = sum(bc[i, :])
        if col_sum > 0:
            bc[i, :] /= float(col_sum)
    return bc, bex, bey


def plot_events_and_response(df, bins, rem, lims):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(str(bins)+' bins' if type(bins) == int else 'Bayesian ('+str(len(bins)-1)+')')
    ax[0].hist2d(df['reco'], df['gen'], bins, range=[lims, lims])
    ax[0].set_title('Gen v Reco')
    ax[0].set_xlabel('reco')
    ax[0].set_ylabel('gen')
    fig.colorbar(ax[0].images[-1], ax=ax[0])

    pc = ax[1].pcolor(rem[1], rem[2], rem[0].T)
    ax[1].axis([rem[1].min(), rem[1].max(), rem[2].min(), rem[2].max()])
    plt.colorbar(pc, ax=ax[1])
    ax[1].set_title('Response Matrix')
    ax[1].set_xlabel('reco')
    ax[1].set_ylabel('gen')


def plot_inv_rem(rem_tuple):
    pinv = linalg.pinv(rem_tuple[0])
    plt.pcolor(rem_tuple[1], rem_tuple[2], pinv.T)
    plt.axis([rem_tuple[1].min(), rem_tuple[1].max(), rem_tuple[2].min(), rem_tuple[2].max()])
    plt.colorbar()
    plt.title('Response Matrix Pseudo-Inverse')
    plt.xlabel('reco')
    plt.ylabel('gen')
    return pinv


def unfold_and_plot(rem, bc, bin_edges):
    inv_rem = linalg.pinv(rem)
    bc_shift = np.dot(inv_rem, bc)
    cov = np.zeros(inv_rem.shape)
    for i in range(inv_rem.shape[0]):
        for j in range(inv_rem.shape[1]):
            for k in range(len(bc)):
                cov[i,j] += inv_rem[i,k]*inv_rem[j,k]*bc[k]
    print np.diag(cov)

    bin_left = bin_edges[:-1]
    bin_width = bin_edges[1:]-bin_edges[:-1]
    #plt.bar(bin_left, bc/bin_width, width=bin_width, alpha=0.5, label='reco')
    plt.bar(bin_left, bc_shift, width=bin_width, alpha=0.5,
            color='red', label='unfolded', yerr=np.sqrt(np.abs(np.diag(cov))), ecolor='r')
    plt.ylim((max(plt.ylim()[0], -plt.ylim()[1]), plt.ylim()[1]))
    plt.legend()
