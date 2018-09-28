# -*- coding: utf-8 -*-
"""
A set of routines for analysing an isolated run


Rok Roskar
University of Zurich

"""

import warnings

import numpy as np
import pynbody
import scipy as sp
from pynbody.analysis.angmom import faceon
from pynbody.analysis.profile import Profile, VerticalProfile

from . import diskfitting, parallel_util
from .parallel_util import interruptible, run_parallel


#try:
#    from IPython.parallel import Client
#    lv = Client().load_balanced_view()
#except :
#    warnings.warn("Parallel interface not able to start -- using serial functions", RuntimeWarning)


def get_rform(sim):
    """

    Determine the formation radius of star particles after reading
    the starlog file and the center of mass information

    """

    import os.path

    base = sim

    while hasattr(base, 'base'):
        base = base.base

    path = os.path.abspath(os.path.dirname(base.filename))
    path += '/../'
    data = np.load(path + 'cofm.npz')

    if hasattr(sim, 'base'):
        up = sim.base
        while hasattr(up, 'base'):
            up = up.base


#    try:
#        for arr in ['x','y','z'] :
#            del(up[arr+'form'])

#    except KeyError:
#        pass

    try:
        sim['posform']
        sim['velform']
        starlog = True
    except KeyError:
        starlog = False
        sim['posform'] = sim['rform']
        sim['posform'].units = sim['x'].units
        sim['velform'] = sim['vform']
        sim['velform'].units = sim['vx'].units

        del (sim['rform'])

    for i, arr in enumerate(['x', 'y', 'z']):
        #spl = sp.interpolate.interp1d(data['times'], data['cofm'][:,i],kind='linear',bounds_error=False)
        #        spl = sp.interpolate.UnivariateSpline(data['times'], data['cofm'][:,i]
        if starlog: pass
        else:
            sim[arr + 'form'] = sim['posform'][:, i]
            sim['v' + arr + 'form'] = sim['velform'][:, i]

        sim[arr + 'form'] -= sp.interp(sim['tform'], data['times'],
                                       data['cofm'][:, i])
        sim['v' + arr + 'form'] -= sp.interp(sim['tform'], data['times'],
                                             data['vcofm'][:, i])

    sim['posform'] = np.array([sim['xform'], sim['yform'], sim['zform']]).T
    sim['velform'] = np.array([sim['vxform'], sim['vyform'], sim['vzform']]).T
    sim['rform'] = np.sqrt(np.sum(sim['posform'][:, 0:2]**2, axis=1))
    sim['jzform'] = sim['xform'] * sim['vyform'] - sim['vxform'] * sim['yform']


@pynbody.snapshot.SimSnap.derived_quantity
def jzmaxr(self):
    """
    Calculate the maximum angular momentum given the star's radius
    """
    from scipy import interp
    top = self
    while hasattr(top, 'base'):
        top = top.base

    prof = pynbody.analysis.profile.Profile(
        top, max='100 kpc', min='.001 kpc', type='log', nbins=100)
    return pynbody.array.SimArray(
        interp(self['rxy'], prof['rbins'], prof['j_circ']),
        units=prof['j_circ'].units)


@pynbody.snapshot.SimSnap.derived_quantity
def jzmaxe(self):
    """
    Calculate the maximum angular momentum given the star's energy
    """
    from scipy import interp
    top = self
    while hasattr(top, 'base'):
        top = top.base

    disk = pynbody.filt.Disc('50 kpc', '500 pc')
    prof = pynbody.analysis.profile.Profile(
        top, nbins=100, type='log', min='0.01 kpc', max='100 kpc')
    return pynbody.array.SimArray(
        interp(self['te'], prof['E_circ'], prof['j_circ']),
        units=prof['j_circ'].units)


def get_cofm(dir='./', filepattern='*/*.0????'):
    """

    Generate a center of mass data file from all the outputs in a run

    **Optional Keywords**:

       *dir*: base directory

       *filepattern*: the file pattern to search for

       *filelist*: list of filenames to process -- if specified, *dir* and
          *filepattern* are ignored

    """

    filelist = glob.glob(filepattern)
    filelist.sort()

    times = pynbody.array.SimArray(np.empty(len(filelist)))
    #times.units = 'Gyr'
    cofms = pynbody.array.SimArray(np.empty((len(filelist), 3)))
    #cofms.units = 'kpc'

    for i, name in enumerate(filelist):
        times[i], cofms[i] = get_cofm_single_file(name)

    np.savez('cofm.npz', cofm=cofms, times=times)


def get_cofm_parallel(dir='./',
                      filepattern='*/*.00???',
                      filelist=None,
                      block=True,
                      procs=pynbody.config['number_of_threads']):
    """

    A parallel version of get_cofm() -- uses the IPython load balanced view
    to farm out the work.

    Generate a center of mass data file from all the outputs in a run

    **Optional Keywords**:

       *dir*: base directory

       *filepattern*: the file pattern to search for

       *filelist*: list of filenames to process -- if specified, *dir* and
          *filepattern* are ignored

    """

    import glob

    if filelist is None:
        filelist = glob.glob(dir + filepattern)

    if len(filelist) == 0:
        raise RuntimeError("No files found matching " + dir + filepattern)

    filelist.sort()

    times = np.empty(len(filelist))
    cofms = np.empty((len(filelist), 3))
    vcofms = np.empty((len(filelist), 3))

    res = run_parallel(get_cofm_single_file, filelist, [], processes=procs)

    if block:
        res = sorted(res)

        for i, x in enumerate(res):
            times[i] = res[i][0]
            cofms[i] = res[i][1]
            vcofms[i] = res[i][2]

        np.savez(dir + 'cofm.npz', cofm=cofms, vcofm=vcofms, times=times)
    else:
        return res, filelist


@interruptible
def get_cofm_single_file(args):
    """

    Return the center of mass of a single file

    **Input**:

       *name*: filename

    """

    name = args

    s = pynbody.load(name)

    time = s.properties['time'].in_units('Gyr')
    cofm = pynbody.analysis.halo.center(s, mode='hyb', retcen=True)
    s.ancestor['pos'] -= cofm
    vcofm = pynbody.analysis.halo.vel_center(s.s, retcen=True)

    return time, np.array(cofm), np.array(vcofm)


def plot_dist_mean(x, y, mass, **kwargs):
    """

    Plots the KDE (or 2D histogram) of the data including points
    showing the mean trend.

    """
    import matplotlib.pylab as plt

    g, xs, ys = pynbody.plot.generic.gauss_kde(x, y, mass=mass, **kwargs)

    range = kwargs.get('x_range', None)

    h1, bins = np.histogram(x, weights=y * mass, range=range)
    h1_mass, bins = np.histogram(x, weights=mass, range=range)

    h1 /= h1_mass
    bins = .5 * (bins[:-1] + bins[1:])

    plt.plot(bins, h1, 'oy')


def plot_means(x, y, mass, range, *args, **kwargs):
    import matplotlib.pylab as plt

    h1, bins = np.histogram(x, weights=y * mass, range=range)
    h1_mass, bins = np.histogram(x, weights=mass, range=range)

    h1 /= h1_mass
    bins = .5 * (bins[:-1] + bins[1:])

    plt.plot(bins, h1, *args, **kwargs)


def one_d_kde(x, weights=None, range=None, gridsize=100):
    """

    generate a 1D weighted KDE

    """

    import scipy as sp
    import numpy as np
    import scipy.sparse

    nx = gridsize

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(x.size)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                             ' as input x & y arrays!')

    # Default extents are the extent of the data
    if range is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = list(map(float, range))

    dx = (xmax - xmin) / (nx - 1)

    xi = np.vstack(x).T
    xi -= xmin
    xi /= dx
    xi = np.floor(xi).T

    grid = sp.sparse.coo_matrix((weights, xi), shape=nx).toarray()

    cov = np.cov(xi)

    scotts_factor = np.power(x.size, -1.0 / 4)

    std_devs = np.diag(np.sqrt(cov))

    # this next line sets the size of the kernel in pixels

    kern_nx = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor**2)

    xx = np.arange(kern_nx, dtype=np.float) - kern_nx/2.0\

    return kern_nx, inv_cov, xx


@interruptible
def single_profile_fits(x):
    from pynbody.analysis.profile import Profile, VerticalProfile
    from fitting import fit_profile, two_sech2, expo

    filename, merger = x

    s = pynbody.load(filename)
    pynbody.analysis.angmom.faceon(s)

    if merger: s = s.s[np.where(s.s['mass'] > .1)[0]]

    p = Profile(s.s, min=0, max=15, nbins=30)
    fit_r, chsq = fit_profile(p, expo, [1e9, 3], 'Msol kpc^-2', 3, 6)

    # make vertical profiles at 1, 2, 3 scalelengths
    pv1 = VerticalProfile(s.s, fit_r[1] * .8, fit_r[1] * 1.2, 3.0, nbins=30)
    pv2 = VerticalProfile(s.s, fit_r[1] * 1.8, fit_r[1] * 2.2, 3.0, nbins=30)
    pv3 = VerticalProfile(s.s, fit_r[1] * 2.8, fit_r[1] * 3.2, 3.0, nbins=30)

    fit_v1, chsq1 = fit_profile(pv1, two_sech2, [0.1, .2, 0.01, .5],
                                'Msol pc^-3', 0, 3)
    fit_v2, chsq2 = fit_profile(pv2, two_sech2, [0.1, .2, 0.01, .5],
                                'Msol pc^-3', 0, 3)
    fit_v3, chsq3 = fit_profile(pv3, two_sech2, [0.1, .2, 0.01, .5],
                                'Msol pc^-3', 0, 3)

    return s.properties['time'].in_units('Gyr'), fit_r, fit_v1, fit_v2, fit_v3


def disk_structure_evolution(flist, merger=False):

    times = np.zeros(len(flist))
    rfits = np.zeros((len(flist), 2))
    v1fits = np.zeros((len(flist), 4))
    v2fits = np.zeros((len(flist), 4))
    v3fits = np.zeros((len(flist), 4))

    res = run_parallel(single_profile_fits, flist, [merger], processes=1)

    for i in range(len(flist)):
        times[i] = res[i][0]
        rfits[i] = res[i][1]
        v1fits[i] = res[i][2]
        v2fits[i] = res[i][3]
        v3fits[i] = res[i][4]

    return times, rfits, v1fits, v2fits, v3fits


def plot_profile_fit(filename, merger, rmin=3, rmax=6):

    s = pynbody.load(filename)
    faceon(s)
    if merger: s = s.s[np.where(s.s['mass'] > .1)[0]]
    p = Profile(s.s, min=0, max=15, nbins=30)
    fit, chsq = fit_profile(p, expo, [1e9, 3], 'Msol kpc^-2', rmin, rmax)

    plt.figure()
    plt.plot(p['rbins'], p['density'].in_units('Msol kpc^-2'))
    overplot_fit(fit, expo)
    print((fit, chsq))
    plt.semilogy()


def get_zrms_grid(s, varx, vary, rmin, rmax, zmin, zmax, gridsize=(20, 20)):
    """
    Produces z_rms values on a grid specified by varx and vary

    """

    #    s.s['dr'] = s.s['rxy']-s.s['rform']

    hist, xs, ys = pynbody.plot.generic.hist2d(
        s.s[varx],
        s.s[vary],
        mass=s.s['mass'],
        make_plot=False,
        gridsize=gridsize)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    zrms = np.zeros((len(xs), len(ys))) * float('Nan')
    zrms_i = np.zeros((len(xs), len(ys))) * float('Nan')
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ind = np.where(
                (s.s[varx] > x - dx / 2) & (s.s[varx] < x + dx / 2) &
                (s.s[vary] > y - dy / 2) & (s.s[vary] < y + dy / 2))[0]

            print((i, j, len(ind)))
            if len(ind) > 100:

                zrms[j, i] = np.sqrt((s.s['z'][ind]**2).sum() / len(ind))
                zrms_i[j, i] = np.sqrt((s.s['zform'][ind]**2).sum() / len(ind))

            else:
                zrms[j, i] = -500
    return hist, zrms, zrms_i, xs, ys


def get_hz_grid(s,
                varx,
                vary,
                rmin,
                rmax,
                zmin,
                zmax,
                gridsize=(10, 10),
                simplefit=True,
                plots=False):
    from fitting import fit_profile, sech2, expo, overplot_fit
    from matplotlib.backends.backend_pdf import PdfPages
    from pynbody.analysis.profile import VerticalProfile

    hist, xs, ys = pynbody.plot.generic.hist2d(
        s.s[varx],
        s.s[vary],
        mass=s.s['mass'],
        make_plot=False,
        gridsize=gridsize)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    print((dx, dy))

    hz = np.zeros((len(xs), len(ys)))
    hr = np.zeros((len(xs), len(ys)))
    num = np.zeros((len(xs), len(ys)))
    hzerr = np.zeros((len(xs), len(ys)))
    hrerr = np.zeros((len(xs), len(ys)))
    rho0 = np.zeros((len(xs), len(ys)))

    if plots:
        pp = PdfPages('vertical_fits.pdf')
        f = plt.figure()

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ind = np.where(
                (s.s[varx] > x - dx / 2) & (s.s[varx] < x + dx / 2) &
                (s.s[vary] > y - dy / 2) & (s.s[vary] < y + dy / 2))[0]

            print((i, j, len(ind)))
            if len(ind) > 100:
                prof = VerticalProfile(s.s[ind], 0, 20, 3.0, nbins=10)

                if simplefit:  # doing the least squares fit

                    fit, chisq = fit_profile(
                        prof, expo,
                        [prof['density'].in_units('Msol pc^-3')[0], .2],
                        'Msol pc^-3', 0, 3)

                    hz[j, i] = fit[1]
                    hr[j, i] = float('Nan')
                    num[j, i] = len(ind)
                    hzerr[j, i] = float('Nan')
                    hrerr[j, i] = float('Nan')
                    rho0[j, i] = fit[0]

                    if plots:
                        plt.plot(prof['rbins'],
                                 prof['density'].in_units('Msol pc^-3'))
                        plt.semilogy()
                        plt.annotate(
                            '%f, %f' % (x, y), (0.5, 0.5),
                            xycoords='figure fraction')
                        overplot_fit(fit, expo)
                        pp.savefig()
                        plt.clf()
                else:  # doing the maximum likelihood fit
                    hr0, hz0, fitnum = diskfitting.two_exp_fit(
                        s.s[ind],
                        rmin=rmin,
                        rmax=rmax,
                        zmin=zmin,
                        zmax=zmax,
                        func=diskfitting.negexpsech)
                    #                    hrerr0, hzerr0 = diskfitting.mcerrors(s.s[ind],[hr0,hz0],rmin=rmin,
                    ###                                                         rmax=rmax,zmin=zmin,zmax=zmax)
                    hz[j, i] = hz0
                    hr[j, i] = hr0
                    hzerr[j, i] = float('Nan')
                    hrerr[j, i] = float('Nan')
                    num[j, i] = fitnum

                    if plots:
                        prof = VerticalProfile(s.s[ind], 0, 20, 3.0, nbins=10)
                        diskfitting.plot_sech_profile(
                            prof, hz0 / 2, xy=(x, y), units='m_p cm^-3')
                        pp.savefig()
                        plt.clf()

            else:
                hz[j, i] = -500

    if plots:
        pp.close()

    return hist, hz, hr, float('Nan'), float('Nan'), hzerr, hrerr, xs, ys, num


def get_hz_grid_parallel(s,
                         varx,
                         vary,
                         rmin,
                         rmax,
                         zmin,
                         zmax,
                         gridsize=(20, 20),
                         ncpu=int(pynbody.config['number_of_threads']),
                         form='sech',
                         get_errors=False):
    from .parallel_util import run_parallel, interruptible

    hist, xs, ys = pynbody.plot.generic.hist2d(
        s.s[varx],
        s.s[vary],
        mass=s.s['mass'],
        make_plot=False,
        gridsize=gridsize)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    print((dx, dy))

    # generate x,y grid

    points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 1, 2).squeeze()

    annulus = pynbody.filt.Disc(rmax, zmax) & ~pynbody.filt.Disc(
        rmin, zmax) & ~pynbody.filt.Disc(rmax, zmin)

    x = np.array(s.s[annulus][varx]).copy()
    y = np.array(s.s[annulus][vary]).copy()
    rxy = np.array(s.s[annulus]['rxy']).copy()
    z = np.array(s.s[annulus]['z']).copy()

    # get the fit

    res = np.array(
        run_parallel(
            fit_single_profile,
            list(points),
            [x, y, rxy, z, dx, dy, rmin, rmax, zmin, zmax, form, get_errors],
            processes=ncpu))

    hr = res[:, 0].reshape(gridsize).T
    hz = res[:, 1].reshape(gridsize).T
    hr2 = res[:, 2].reshape(gridsize).T
    hz2 = res[:, 3].reshape(gridsize).T
    hrerr = res[:, 4].reshape(gridsize).T
    hzerr = res[:, 5].reshape(gridsize).T
    fitnum = res[:, 6].reshape(gridsize).T

    return hist, hz, hr, hz2, hr2, hzerr, hrerr, xs, ys, fitnum


@interruptible
def fit_single_profile(a):
    point, x, y, rxy, z, dx, dy, rmin, rmax, zmin, zmax, form, get_errors = a

    if form == 'sech':
        func = diskfitting.negexpsech
        errfunc = diskfitting.exp_sech_likelihood
    else:
        func = diskfitting.neg2expl
        errfunc = diskfitting.twoexp_likelihood

    px, py = point
    print(('point = ', point))

    ind = np.where((x > px - dx / 2) & (x < px + dx / 2) & (y > py - dy / 2) &
                   (y < py + dy / 2))[0]

    fitnum = len(ind)

    if fitnum > 100:
        hr, hz, fitnum = diskfitting.two_exp_fit_simple(
            np.array(rxy[ind]),
            np.array(z[ind]),
            rmin,
            rmax,
            zmin,
            zmax,
            func=func)
        #        hz = np.median(np.abs(z[ind]))
        #        hr = float('Nan')

        if get_errors:
            hr2, hz2, hrerr, hzerr = diskfitting.mcerrors_simple(
                np.array(rxy[ind]),
                np.array(z[ind]),
                hr,
                hz,
                rmin,
                rmax,
                zmin,
                zmax,
                nwalkers=6,
                func=errfunc)
        else:
            hz2 = float('Nan')
            hr2 = float('Nan')
            hzerr = float('Nan')
            hrerr = float('Nan')
    else:
        hr = float('Nan')
        hz = float('Nan')
        hr2 = float('Nan')
        hz2 = float('Nan')
        hrerr = float('Nan')
        hzerr = float('Nan')

    if form == 'sech':
        hz /= 2.0
        hz2 /= 2.0
        hzerr /= 2.0

    return hr, hz, hr2, hz2, hrerr, hzerr, fitnum


#@interruptible
#def fit_single_errors(a):


def get_vdisp_grid(s, varx, vary, gridsize=(10, 10)):

    s.s['dr'] = s.s['rxy'] - s.s['rform']

    hist, xs, ys = pynbody.plot.generic.hist2d(
        s.s[varx],
        s.s[vary],
        mass=s.s['mass'],
        make_plot=False,
        gridsize=gridsize)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    vdisp_r = np.zeros((len(xs), len(ys))) * float('Nan')
    vdisp_z = np.zeros((len(xs), len(ys))) * float('Nan')

    vdisp_r_i = np.zeros((len(xs), len(ys))) * float('Nan')
    vdisp_z_i = np.zeros((len(xs), len(ys))) * float('Nan')

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ind = np.where(
                (s.s[varx] > x - dx / 2) & (s.s[varx] < x + dx / 2) &
                (s.s[vary] > y - dy / 2) & (s.s[vary] < y + dy / 2))[0]

            if len(ind) > 100:

                vdisp_r[j, i] = np.std(s.s['vr'][ind])
                vdisp_z[j, i] = np.std(s.s['vz'][ind])
                vdisp_z_i[j, i] = np.std(s.s['vzform'][ind])

    return hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys


def plot_2D_grid(s, varx, vary, varz, gridsize=(10, 10)):
    """
    Produces a 2D plot binning the property varz on a grid specified
    by varx and vary.

    """

    hist, xs, ys = pynbody.plot.generic.hist2d(
        s.s[varx],
        s.s[vary],
        mass=s.s['mass'],
        make_plot=False,
        gridsize=gridsize)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    zvals = np.zeros(gridsize) * np.float('Nan')

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ind = np.where(
                (s.s[varx] > x - dx / 2) & (s.s[varx] < x + dx / 2) &
                (s.s[vary] > y - dy / 2) & (s.s[vary] < y + dy / 2))[0]

            print((i, j, len(ind)))
            if len(ind) > 100:

                zrms[j, i] = np.sqrt((s.s['z'][ind]**2).sum() / len(ind))
                zrms_i[j, i] = np.sqrt((s.s['zform'][ind]**2).sum() / len(ind))

            else:
                zrms[j, i] = -500
    return hist, zrms, zrms_i, xs, ys


def plot_age_velocity_relation(s,
                               limits=pynbody.filt.SolarNeighborhood(
                                   7.5, 8.5, .2)):
    import matplotlib.pylab as plt
    from scipy import polyfit

    prof = pynbody.analysis.profile.Profile(
        s.s[limits], calc_x=lambda x: x['age'], type='log', nbins=10, min=1)

    plt.plot(
        prof['rbins'],
        np.sqrt(prof['vr_disp']**2 + prof['vt_disp']**2 + prof['vz_disp']**2),
        'k-',
        label=r'$\sigma_{tot}$',
        linewidth=2)
    plt.plot(
        prof['rbins'],
        prof['vr_disp'],
        'k--',
        label=r'$\sigma_R$',
        linewidth=2)
    plt.plot(
        prof['rbins'],
        prof['vt_disp'],
        'k-.',
        label=r'$\sigma_{\phi}$',
        linewidth=2)
    plt.plot(
        prof['rbins'], prof['vz_disp'], 'k:', label=r'$\sigma_z$', linewidth=2)

    plt.loglog()

    plt.xlabel('Age [Gyr]')
    plt.ylabel('$\sigma$ [km/s]')
    plt.legend(loc='upper left', prop=dict(size='small'))

    # fits
    fittot = polyfit(
        np.log10(prof['rbins']),
        np.log10(
            np.sqrt(prof['vr_disp']**2 + prof['vt_disp']**2 +
                    prof['vz_disp']**2)), 1)
    fitr = polyfit(np.log10(prof['rbins']), np.log10(prof['vr_disp']), 1)
    fitt = polyfit(np.log10(prof['rbins']), np.log10(prof['vt_disp']), 1)
    fitz = polyfit(np.log10(prof['rbins']), np.log10(prof['vz_disp']), 1)

    plt.plot(prof['rbins'], 10**fitr[1] * prof['rbins']**fitr[0], 'r--')
    plt.plot(prof['rbins'], 10**fitt[1] * prof['rbins']**fitt[0], 'r--')
    plt.plot(prof['rbins'], 10**fitz[1] * prof['rbins']**fitz[0], 'r--')

    print(fittot)
    print(fitr)
    print(fitt)
    print(fitz)


def plot_zrms_vs_r(s, rs=[2, 4, 6, 8, 10, 12, 14], ages=None):
    import matplotlib.pylab as plt

    pynbody.analysis.angmom.faceon(s)

    f, ax = plt.subplots()

    if ages is None:
        ages = np.array([[s.s['age'].min(), s.s['age'].max()]])
    else:
        ages = np.array(ages)

    print(ages)

    zrms = np.zeros((len(rs), len(ages)))

    for i, r in enumerate(rs):

        #        p = pynbody.analysis.profile.VerticalProfile(s.s, r-.5,r+.5,3.0,nbins=30)
        if ages is not None:
            for j, age in enumerate(ages):
                ind = np.where((s.s['rxy'] < r + .5) & (s.s['rxy'] > r - .5) &
                               (s.s['age'] > age.min()) &
                               (s.s['age'] < age.max()))[0]
                zrms[i, j] = np.sqrt((s.s['z'][ind]**2).sum() / len(ind))
                print((len(ind), age.min(), age.max()))

    for i in range(len(ages)):
        ax.plot(
            rs,
            zrms[:, i],
            'o',
            label='%.1f $<$ age $<$ %.1f' % (ages[i].min(), ages[i].max()))

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('$z_{rms}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 15)
    ax.legend(loc='upper left', frameon=False)
    return zrms


def compare_zrms_vs_r(slist,
                      names,
                      rs=[2, 4, 6, 8, 10, 12, 14],
                      agefilt=pynbody.filt.BandPass('age', 0, 10),
                      ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for j, s in enumerate(slist):

        pynbody.analysis.angmom.faceon(s)

        zrms = np.zeros(len(rs))

        for i, r in enumerate(rs):
            rfilt = pynbody.filt.BandPass('rxy', r - .5, r + .5)
            sfilt = s.s[agefilt & rfilt]
            zrms[i] = np.sqrt((sfilt['z']**2).sum() / len(sfilt))

        ax.plot(rs, zrms, 'o', label=names[j])

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('$z_{rms}$')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 15)
    ax.legend(loc='upper left')
