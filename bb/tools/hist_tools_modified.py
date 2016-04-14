import warnings

import numpy as np

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width

#from bb_poly import bayesian_blocks
from bayesian_blocks_modified import bayesian_blocks
from fill_between_steps import fill_between_steps


def hist(x, bins=10, range=None, fitness='poly_events', gamma = None, p0=0.05, *args, **kwargs):
    """Enhanced histogram

    This is a histogram function that enables the use of more sophisticated
    algorithms for determining bins.  Aside from the `bins` argument allowing
    a string specified how bins are computed, the parameters are the same
    as pylab.hist().

    Parameters
    ----------
    x : array_like
        array of data to be histogrammed

    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scott' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    range : tuple or None (optional)
        the minimum and maximum range for the histogram.  If not specified,
        it will be (x.min(), x.max())

    ax : Axes instance (optional)
        specify the Axes on which to draw the histogram.  If not specified,
        then the current active axes will be used.

    **kwargs :
        other keyword arguments are described in pylab.hist().
    """
    if isinstance(bins, str) and "weights" in kwargs:
        warnings.warn("weights argument is not supported: it will be ignored.")
        kwargs.pop('weights')
        weights = None
    elif "weights" in kwargs:
        weights = kwargs['weights']
    else:
        weights = None

    x = np.asarray(x)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        del kwargs['ax']
    else:
        # import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        ax = plt.gca()

    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if (range is not None and (bins in ['blocks',
                                        'knuth', 'knuths',
                                        'scott', 'scotts',
                                        'freedman', 'freedmans'])):
        x = x[(x >= range[0]) & (x <= range[1])]

    if bins in ['block','blocks']:
        bins = bayesian_blocks(t=x,fitness=fitness,p0=p0,gamma=gamma)
    elif bins in ['knuth', 'knuths']:
        dx, bins = knuth_bin_width(x, True, disp=False)
    elif bins in ['scott', 'scotts']:
        dx, bins = scotts_bin_width(x, True)
    elif bins in ['freedman', 'freedmans']:
        dx, bins = freedman_bin_width(x, True)
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)

    if 'scale' in kwargs and kwargs['scale'] == None:
        kwargs.pop('scale')
    elif 'scale' in kwargs:
        scale = kwargs.pop('scale')
        if 'normed' in kwargs:
            normed = kwargs.pop('normed')
        else:
            normed = False
        bin_content, bins = np.histogram(x,bins,range,weights=weights,density=normed)
        #return fill_between_steps(ax, bins, bin_content*scale,0, step_where='pre', **kwargs)
        width = bins[1:]-bins[:-1]
        if 'histtype' in kwargs:
            histtype = kwargs.pop('histtype')
            if histtype=='bar':
                pass
            elif histtype=='stepfilled':
                kwargs['linewidth']=0
        else: histtype='bar'

        if histtype=='step':
            ax.step(bins[0:2], bin_content[0:2]*scale, where='post', **kwargs)
            kwargs['label']=None
            return ax.step(bins[1:], bin_content*scale, where='pre', **kwargs)
        else:
            return ax.bar(bins[:-1],bin_content*scale,width,**kwargs)

    return ax.hist(x, bins, range, **kwargs)
