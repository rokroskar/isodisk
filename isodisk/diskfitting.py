import math

import emcee
import numpy as np
import pynbody
import pynbody.filt as f
import pynbody.units as un
import scipy.optimize as opt
from scipy import stats


# see Bovy, Rix, Liu, Hogg, Beers + Lee 2012, ApJ 753, 148
# though this is much simpler since we don't have selection
# function to deal with
def two_sech_likelihood(x=[1.0, 1.0, 1.0], zpos=None, zmin=0.0, zmax=3.0):

    hz1 = x[0]
    if hz1 <= 0.1 or hz1 > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    hz2 = x[1]
    if hz2 <= 0.1 or hz2 > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    f = x[2]
    if f <= 1e-5 or f > 1.0:
        return -(np.finfo(np.dtype(np.float64)).max)


#    rho1 = x[2]
#    if rho1 <= 1e-5 or rho1 > 1e5: #So it does not go nuts
#       return -(np.finfo(np.dtype(np.float64)).max)

#    rho2 = x[3]
#    if rho1 <= 1e-5 or rho1 > 1e5: #So it does not go nuts
#       return -(np.finfo(np.dtype(np.float64)).max)

    if isinstance(zmin, un.UnitBase):
        zmin = zmin.in_units('kpc')
    if isinstance(zmax, un.UnitBase):
        zmax = zmax.in_units('kpc')

    norm_int = 4 * math.pi * (
        hz1 * (1.0 - f) * (math.tanh(zmax / hz1) - math.tanh(zmin / hz1)) +
        f * hz2 * (math.tanh(zmax / hz2) - math.tanh(zmin / hz2)))

    return np.sum((np.log(
        (1.0 - f) * np.cosh(np.abs(zpos) / hz1)**-2 +
        f * np.cosh(np.abs(zpos) / hz2)**-2))) - len(zpos) * np.log(norm_int)


def sech2_likelihood(x=1.0, zpos=None, zmin=0.0, zmax=3.0):
    hz = x

    if hz <= 0.1 or hz > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    norm_int = 4 * math.pi * (hz *
                              (math.tanh(zmax / hz) - math.tanh(zmin / hz)))

    return np.sum((np.log(np.cosh(np.abs(zpos) / hz)**
                          -2))) - len(zpos) * np.log(norm_int)


def exp_likelihood(x=1.0, zpos=None, zmin=0.0, zmax=3.0):
    hz = x
    if hz <= 0.1 or hz > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    norm_int = (
        4.0 * math.pi * hz * (-np.exp(-zmax / hz) + np.exp(-zmin / hz)))

    return np.sum(-zpos / hz) - len(zpos) * np.log(norm_int)


def double_exp_likelihood(x=[1.0, 1.0, 1.0], zpos=None, zmin=0.0, zmax=3.0):

    hz1 = x[0]
    if hz1 <= 0.1 or hz1 > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    hz2 = x[1]
    if hz2 <= 0.1 or hz2 > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    f = x[2]
    if f <= 1e-5 or f > 1.0:
        return -(np.finfo(np.dtype(np.float64)).max)


#    rho1 = x[2]
#    if rho1 <= 1e-5 or rho1 > 1e5: #So it does not go nuts
#       return -(np.finfo(np.dtype(np.float64)).max)

#    rho2 = x[3]
#    if rho1 <= 1e-5 or rho1 > 1e5: #So it does not go nuts
#       return -(np.finfo(np.dtype(np.float64)).max)

    if isinstance(zmin, un.UnitBase):
        zmin = zmin.in_units('kpc')
    if isinstance(zmax, un.UnitBase):
        zmax = zmax.in_units('kpc')

    norm_int = 4 * math.pi * (
        (1.0 - f) * hz1 * (-np.exp(-zmax / hz1) + np.exp(-zmin / hz1)) +
        f * hz2 * (-np.exp(-zmax / hz2) + np.exp(-zmin / hz2)))

    return np.sum(
        np.log((1.0 - f) * np.exp(-np.abs(zpos) / hz1) +
               f * np.exp(-np.abs(zpos) / hz2))) - len(zpos) * np.log(norm_int)


def exp_sech_likelihood(x=[1.0, 1.0],
                        rpos=None,
                        zpos=None,
                        rmin=4.0,
                        rmax=10.0,
                        zmin=0.0,
                        zmax=3.0):

    hr = x[0]
    if hr <= 0.1 or hr > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    hz = x[1]
    if hz <= 0.1 or hz > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    if isinstance(rmin, un.UnitBase):
        rmin = rmin.in_units('kpc')
    if isinstance(rmax, un.UnitBase):
        rmax = rmax.in_units('kpc')
    if isinstance(zmin, un.UnitBase):
        zmin = zmin.in_units('kpc')
    if isinstance(zmax, un.UnitBase):
        zmax = zmax.in_units('kpc')

    norm_int = (4.0 * math.pi * hr * hz *
                (-np.exp(-rmax / hr) * (hr + rmax) + np.exp(-rmin / hr) *
                 (hr + rmin)) * (math.tanh(zmax / hz) - math.tanh(zmin / hz)))

    return np.sum((-rpos / hr + np.log(np.cosh(np.abs(zpos) / hz)**
                                       -2))) - len(rpos) * np.log(norm_int)


def twoexp_likelihood(x=[1.0, 1.0],
                      rpos=None,
                      zpos=None,
                      rmin=4.0,
                      rmax=10.0,
                      zmin=0.0,
                      zmax=3.0):

    hr = x[0]
    if hr <= 0.1 or hr > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    hz = x[1]
    if hz <= 0.1 or hz > 100.:  #So it does not go nuts
        return -(np.finfo(np.dtype(np.float64)).max)

    if isinstance(rmin, un.UnitBase):
        rmin = rmin.in_units('kpc')
    if isinstance(rmax, un.UnitBase):
        rmax = rmax.in_units('kpc')
    if isinstance(zmin, un.UnitBase):
        zmin = zmin.in_units('kpc')
    if isinstance(zmax, un.UnitBase):
        zmax = zmax.in_units('kpc')

    norm_int = (4.0 * math.pi * hr * hz * (
        (np.exp(-rmax / hr) * (hr + rmax) - np.exp(-rmin / hr) *
         (hr + rmin)) * (np.exp(-zmax / hz) - np.exp(-zmin / hz))))

    return np.sum(
        (-rpos / hr - np.abs(zpos) / hz)) - len(rpos) * np.log(norm_int)


def neg2expl(*a):
    # return negative of likelihood so that minimization scheme maximize
    # likelihood
    return -1 * (two_exp_likelihood(*a))


def negexpsech(*a):
    return -1 * (exp_sech_likelihood(*a))


def negtwosech(*a):
    return -1 * (two_sech_likelihood(*a))


def negtwoexp(*a):
    return -1 * (double_exp_likelihood(*a))


def negexp(*a):
    return -1 * (exp_likelihood(*a))


def negsech2(*a):
    return -1 * (sech2_likelihood(*a))


def two_exp_fit(sim,
                rmin='4 kpc',
                rmax='10 kpc',
                zmin='0 kpc',
                zmax='4 kpc',
                func=neg2expl):

    if isinstance(rmin, str): rmin = un.Unit(rmin)
    if isinstance(rmax, str): rmax = un.Unit(rmax)
    if isinstance(zmin, str): zmin = un.Unit(zmin)
    if isinstance(zmax, str): zmax = un.Unit(zmax)

    annulus = f.Disc(rmax, zmax) & ~f.Disc(rmin, zmax) & ~f.Disc(rmax, zmin)
    fitnum = len(sim.s[annulus])
    print("Fitting %d stars" % (fitnum))
    if fitnum > 100:
        hr, z0 = opt.fmin_powell(
            func, [1.0, 1.0],
            args=(sim.s[annulus]['rxy'].in_units('kpc'),
                  sim.s[annulus]['z'].in_units('kpc'), rmin, rmax, zmin, zmax))
        return hr, z0, fitnum
    else:
        return float('NaN'), float('NaN'), fitnum


def two_exp_fit_simple(r, z, rmin, rmax, zmin, zmax, func=neg2expl):
    fitnum = len(r)
    if fitnum > 100:
        hr, z0 = opt.fmin_powell(
            func, [1.0, 1.0], args=(r, z, rmin, rmax, zmin, zmax))
        return hr, z0, fitnum
    else:
        return float('NaN'), float('NaN'), fitnum


def two_comp_zfit_simple(z, zmin=0, zmax=3, p0=[1.0, 1.0, .5],
                         func=negtwosech):
    zs = z[np.where((z >= zmin) & (z < zmax))[0]]
    fitnum = len(zs)
    if fitnum > 100:
        res = opt.fmin_powell(func, p0, args=(zs, zmin, zmax), disp=0)
        return res, fitnum
    else:
        return float('NaN'), float('Nan'), fitnum


def single_fit(z, zmin=0, zmax=3, p0=[1.0], func=negsech2):
    fitnum = len(z)
    if fitnum > 100:
        res = opt.fmin_powell(func, p0, args=(z, zmin, zmax))
        return res, fitnum
    else:
        return float('NaN'), float('Nan'), fitnum




def mcerrors(sim,
             initparams,
             rmin='4 kpc',
             rmax='10 kpc',
             zmin='0 kpc',
             zmax='4 kpc'):
    if isinstance(rmin, str): rmin = un.Unit(rmin)
    if isinstance(rmax, str): rmax = un.Unit(rmax)
    if isinstance(zmin, str): zmin = un.Unit(zmin)
    if isinstance(zmax, str): zmax = un.Unit(zmax)

    annulus = f.Disc(rmax, zmax) & ~f.Disc(rmin, zmax) & ~f.Disc(rmax, zmin)

    nwalkers, ndim = 6, 2
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        twoexp_likelihood,
        args=(np.array(sim.s[annulus]['rxy'].in_units('kpc')),
              np.array(sim.s[annulus]['z'].in_units('kpc')), rmin, rmax, zmin,
              zmax))

    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sampler.run_mcmc(p0, 1000)
    #import pdb; pdb.set_trace()
    return np.std(sampler.flatchain[:, 0]), np.std(
        sampler.flatchain[:, 1] / 2.0)


def mcerrors_simple_singlevar(z,
                              p0=[1.0, 1.0, 1.0],
                              zmin=0,
                              zmax=3,
                              nwalkers=8,
                              func=two_sech_likelihood):
    ndim = 3

    hz1, hz2, f = p0

    sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=(z, zmin, zmax))

    #    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
    #   print 'trying with %f %f'%(hr,hz)
    p0 = [[
        np.random.normal(hz1, .1),
        np.random.normal(hz2, .1),
        np.random.normal(f, .1)
    ] for i in range(nwalkers)]
    #p0 = sampler.sampleBall([hr,hz],[hr/10.0, hz/10.0],nwalkers)
    # following the 50-D gaussian example on the emcee page...
    # "burn-in"
    pos, prob, state = sampler.run_mcmc(p0, 100)
    # reset to clear the samples
    sampler.reset()
    # re-run starting from the final position of the chain
    sampler.run_mcmc(p0, 1000)

    #import pdb; pdb.set_trace()
    return np.median(sampler.flatchain[:,0]), np.median(sampler.flatchain[:,1]), \
        np.median(sampler.flatchain[:,2]), \
        np.std(sampler.flatchain[:,0]), np.std(sampler.flatchain[:,1]), \
        np.std(sampler.flatchain[:,2])


def mcerrors_simple(r,
                    z,
                    hr,
                    hz,
                    rmin,
                    rmax,
                    zmin,
                    zmax,
                    nwalkers=6,
                    func=twoexp_likelihood):
    ndim = 2

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, func, args=(r, z, rmin, rmax, zmin, zmax))

    #    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
    #   print 'trying with %f %f'%(hr,hz)
    p0 = [[np.random.normal(2.5, 1),
           np.random.normal(.5, .1)] for i in range(nwalkers)]
    #p0 = sampler.sampleBall([hr,hz],[hr/10.0, hz/10.0],nwalkers)
    # following the 50-D gaussian example on the emcee page...
    # "burn-in"
    pos, prob, state = sampler.run_mcmc(p0, 100)
    # reset to clear the samples
    sampler.reset()
    # re-run starting from the final position of the chain
    sampler.run_mcmc(pos, 100, rstate0=state)

    #import pdb; pdb.set_trace()
    return np.median(sampler.flatchain[:,0]), np.median(sampler.flatchain[:,1]), \
        np.std(sampler.flatchain[:,0]), np.std(sampler.flatchain[:,1])


def plot_sech_profile(profile,
                      fith=1.0,
                      outfig=False,
                      units='m_p cm^-2',
                      title=False,
                      zmin=0.0,
                      zmax=4.0,
                      xy=(0, 0)):
    import matplotlib.pyplot as plt

    logN = np.log10(profile['density'].in_units(units))
    good = np.isfinite(logN)
    rad = profile['rbins'].in_units('kpc')
    if isinstance(zmin, str): zmin = un.Unit(zmin)
    if isinstance(zmax, str): zmax = un.Unit(zmax)

    # Do simple fit to get normalization
    #    expfit = np.polyfit(np.array(rad[good]), np.array(logN[good]), 1)
    # 1.0857 is how many magnitudes a 1/e decrease is
    #h=-1.0857/expfit[0]
    #   N_0=expfit[1]
    N_0 = np.log10(
        np.sum(profile['density'].in_units(units)) / 2 * fith *
        (math.tanh(zmax / (2 * fith)) - math.tanh(zmin / (2 * fith))))

    print("N_0: %g" % N_0)
    plt.errorbar(
        rad[good],
        logN[good],
        yerr=np.log10(profile['density'][good].in_units(units)) / np.sqrt(
            profile['n'][good]),
        fmt='o')
    plt.plot(
        rad,
        N_0 + np.log10(np.cosh(rad / (2.0 * fith))**-2),
        label='h: %2.1f kpc; $N_0$: %2.1f' % (fith, N_0),
        linestyle='dashed')
    if title: plt.title(title)
    plt.xlabel('distance [kpc]')
    plt.ylabel('log$_{10}$(surface density [$' +
               pynbody.units.Unit(units).latex() + '$])')
    plt.legend(loc=0)
    plt.title('x = %.2f y = %.2f' % (xy[0], xy[1]))

    if outfig:
        plt.savefig(outfig + '.png')


def plot_profile(profile,
                 fith=1.0,
                 xy=(0, 0),
                 outfig=False,
                 units='m_p cm^-2',
                 title=False,
                 rmin='4 kpc',
                 rmax='10 kpc'):
    import matplotlib.pyplot as plt

    if isinstance(rmin, str): rmin = un.Unit(rmin)
    if isinstance(rmax, str): rmax = un.Unit(rmax)

    logN = np.log10(profile['density'].in_units(units))
    good = np.isfinite(logN)
    rad = profile['rbins'].in_units('kpc')

    # Do simple fit to get normalization
    expfit = np.polyfit(np.array(rad[good]), np.array(logN[good]), 1)
    # 1.0857 is how many magnitudes a 1/e decrease is
    h = -1.0857 / expfit[0]
    N_0 = expfit[1]
    #N_0 = np.log10(np.sum(vprofile['density'].in_units(units)) /
    #               (4.0*hz*(math.tanh(zmax/(2.0*hz))-math.tanh(zmin/(2.0*hz))))/
    #               (2.0*math.pi*hr*(-math.exp(-rmax/hr)*(hr + rmax) +
    #                                 math.exp(-rmin/hr)*(hr + rmin))))
    #N_0=np.log10(np.sum(profile['density'].in_units(units)) /
    #             2.0*math.pi*fith*(-np.exp(-rmax/fith)*(fith + rmax) +
    #                                np.exp(-rmin/fith)*(fith + rmin)))

    plt.errorbar(
        rad[good],
        logN[good],
        yerr=np.log10(profile['density'][good].in_units(units)) / np.sqrt(
            profile['n'][good]),
        fmt='o')
    plt.plot(
        rad,
        N_0 + np.log10(np.exp(-rad / fith)),
        label='h: %2.1f kpc; $N_0$: %2.1f' % (fith, N_0),
        linestyle='dashed')
    if title: plt.title(title)
    plt.xlabel('distance [kpc]')
    plt.ylabel('log$_{10}$(surface density [$' +
               pynbody.units.Unit(units).latex() + '$])')
    plt.legend(loc=0)
    plt.title('x = %.2f y = %.2f' % (xy[0], xy[1]))
    if outfig:
        plt.savefig(outfig + '.png')


def plot_two_profiles(vprofile,
                      rprofile,
                      hz=1.0,
                      hr=1.0,
                      outfig=False,
                      units='m_p cm^-2',
                      title=False,
                      rmin='4 kpc',
                      rmax='10 kpc',
                      zmin='0 kpc',
                      zmax='4 kpc'):
    import matplotlib.pyplot as plt

    if outfig: plt.ioff()
    else: plt.ion()
    f, ax = plt.subplots(1, 2)

    if isinstance(rmin, str): rmin = un.Unit(rmin).in_units('kpc')
    if isinstance(rmax, str): rmax = un.Unit(rmax).in_units('kpc')
    if isinstance(zmin, str): zmin = un.Unit(zmin).in_units('kpc')
    if isinstance(zmax, str): zmax = un.Unit(zmax).in_units('kpc')

    vrad = vprofile['rbins'].in_units('kpc')
    vpmass = vprofile['mass'].in_units('Msol')

    dz = vprofile['rbins'].in_units('kpc')[1] - vprofile['rbins'].in_units(
        'kpc')[0]
    vbinsize = math.pi * (rmax * rmax - rmin * rmin) * dz

    vN_0 = (np.sum(vpmass / vbinsize * dz) /
            (2.0 * hz *
             (math.tanh(zmax / (2.0 * hz)) - math.tanh(zmin / (2.0 * hz)))))

    ax[0].errorbar(
        vrad,
        vpmass / vbinsize,
        yerr=vpmass / vbinsize / np.sqrt(vprofile['n']),
        fmt='o')
    ax[0].semilogy(
        vrad,
        vN_0 * (np.cosh(vrad / (2.0 * hz))**-2),
        label='h: %2.1f kpc; $N_0$: %2.1f' % (hz, vN_0),
        linestyle='dashed')
    if title: ax[0].set_title(title)
    ax[0].set_xlabel('Z [kpc]')
    ax[0].set_ylabel('Density [M$_\odot$ kpc$^{-3}$]')
    ax[0].legend(loc=0)

    rrad = rprofile['rbins'].in_units('kpc')
    rpmass = rprofile['mass'].in_units('Msol')

    rbinsize = rprofile._binsize.in_units('kpc^2') * (zmax - zmin)
    rN_0 = (np.sum(rpmass / rbinsize * rprofile['dr'].in_units('kpc')) /
            (2.0 * math.pi * hr *
             (-math.exp(-rmax / hr) * (hr + rmax) + math.exp(-rmin / hr) *
              (hr + rmin))))

    ax[1].errorbar(
        rrad,
        rpmass / rbinsize,
        yerr=rpmass / rbinsize / np.sqrt(rprofile['n']),
        fmt='o')
    ax[1].semilogy(
        rrad,
        rN_0 * (np.exp(-rrad / hr)),
        label='h: %2.1f kpc; $N_0$: %2.1f' % (hr, rN_0),
        linestyle='dashed')
    ax[1].set_xlabel('R [kpc]')
    ax[1].set_ylabel('Density [M$_\odot$ kpc$^{-3}$]')
    ax[1].legend(loc=0)
    if outfig:
        plt.savefig(outfig + '.png')
    plt.clf()
    plt.close()


def velocity_dispersion(sim, name='vz'):
    sq_mean = (sim[name].in_units('km s^-1')**2 *
               sim['mass']).sum() / sim['mass'].sum()
    mean_sq = ((sim[name].in_units('km s^-1') * sim['mass']).sum() /
               sim['mass'].sum())**2
    return math.sqrt(sq_mean - mean_sq)


def block_histogram(qty='rexp',
                    file='01024/g1536.01024.z0metbinagedecomp.dat',
                    qtytitle='$h_z$'):
    import pickle
    d = pickle.load(open(file))
    minofe = np.min(d['ofe'])
    maxofe = np.max(d['ofe']) + d['ofestep']
    minfeh = np.min(d['feh'])
    maxfeh = np.max(d['feh']) + d['fehstep']

    nofe = (maxofe - minofe) / d['ofestep']
    nfeh = (maxfeh - minfeh) / d['fehstep']
    arr = np.zeros((nofe, nfeh))
    arr[:, :] = np.nan
    for io, ofe in enumerate(d['ofe']):
        feh = d['feh'][io]
        iofe = (ofe - minofe) / d['ofestep']
        ifeh = (feh - minfeh) / d['fehstep']
        arr[iofe, ifeh] = d[qty][io]

    im = plt.imshow(
        arr[::-1, :],
        extent=(minfeh, maxfeh, minofe, maxofe),
        aspect=(maxfeh - minfeh) / (maxofe - minofe),
        vmin=0.2,
        vmax=1.2,
        interpolation='nearest')
    plt.xlabel('[Fe/H]')
    plt.ylabel('[O/Fe]')
    plt.colorbar().set_label(qtytitle)




def sample_sech2(scale=1., size=1):
    out = np.zeros(size)
    nsamples = 0
    while nsamples < size:
        exps = stats.expon.rvs(scale=scale / 2., size=size - nsamples)
        comp = 1. / np.cosh(exps / scale)**2. / 4. * np.exp(exps / scale * 2.)
        indx = (stats.uniform.rvs(size=size - nsamples) < comp)
        nnewsamples = np.sum(indx)
        if nnewsamples > 0:
            newsamples = exps[indx]
            out[nsamples:nsamples + nnewsamples] = newsamples
            nsamples += nnewsamples
    return out


def sample_two_sech2(scale1=1.0, scale2=2.0, f=0.5, size=10000):
    out = np.zeros(size)
    nsamples = 0

    while nsamples < size:
        exps = stats.expon.rvs(scale=scale2 / 2., size=size - nsamples)
        #exps = stats.uniform.rvs(scale=5,size=size-nsamples)
        comp = two_sech2(
            exps, scale1=scale1, scale2=scale2,
            f=f) / (np.exp(-exps / scale2 * 2.) * 100)
        #        import pdb; pdb.set_trace()
        indx = (stats.uniform.rvs(size=size - nsamples) < comp)
        nnewsamples = np.sum(indx)
        if nnewsamples > 0:
            newsamples = exps[indx]
            out[nsamples:nsamples + nnewsamples] = newsamples
            nsamples += nnewsamples
    return out


def two_sech2(x, scale1=1.0, scale2=2.0, f=0.5):
    return (1 - f) * np.cosh(-x / scale1)**-2 + f * np.cosh(-x / scale2)**-2


#def estimate_error(fit,func=two_sech_fit_simple,ntimes=100):
#    vals = np.zeros((ntimes,len(fit)))
