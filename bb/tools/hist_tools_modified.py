from __future__ import division
import warnings
from numbers import Number
import numpy as np
import matplotlib

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width

#from bb_poly import bayesian_blocks
from bayesian_blocks_modified import bayesian_blocks
from fill_between_steps import fill_between_steps


def hist(x, bins=10, range=None, fitness='events', gamma = None, p0=0.05, *args, **kwargs):
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
        bin_content, bins, patches = ax.hist(x, bins, range, **kwargs)
        bin_error = np.sqrt(bin_content)
        if isinstance(patches[0], matplotlib.patches.Rectangle):
            if scale == 'binwidth':
                for i, bc in enumerate(bin_content):
                    width = (bins[i+1]-bins[i])
                    bin_content[i] /= width
                    bin_error[i] /=width
                    plt.setp(patches[i], 'height', patches[i].get_height()/width)
            elif isinstance(scale, Number):
                for i, bc in enumerate(bin_content):
                    bin_content[i] *= scale
                    bin_error[i] *= scale
                    plt.setp(patches[i], 'height', patches[i].get_height()*scale)
            else:
                warnings.warn("scale argument value `", scale, "` not supported: it will be ignored.")

        elif isinstance(patches[0], matplotlib.patches.Polygon):
            xy = patches[0].get_xy()
            j = 0
            if scale == 'binwidth':
                for i, bc in enumerate(bin_content):
                    width = (bins[i+1]-bins[i])
                    bin_content[i] /= width
                    bin_error[i] /= width
                    xy[j+1,1] = bin_content[i]
                    xy[j+2,1] = bin_content[i]
                    j+=2
                    #plt.setp(patches[i], 'height', patches[i].get_height()/width)
            elif isinstance(scale, Number):
                for i, bc in enumerate(bin_content):
                    bin_content[i] *= scale
                    bin_error[i] *= scale
                    xy[j+1,1] = bin_content[i]
                    xy[j+2,1] = bin_content[i]
                    j+=2
            else:
                warnings.warn("scale argument value `", scale, "` not supported: it will be ignored.")
            plt.setp(patches[0], 'xy', xy)

        ax.relim()
        ax.autoscale_view(False,False,True)
        return bin_content, bins, patches

    return ax.hist(x, bins, range, **kwargs)
