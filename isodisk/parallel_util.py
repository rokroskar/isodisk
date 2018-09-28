import pynbody
import multiprocessing
import numpy as np
from functools import wraps

def run_parallel(func, single_args, repeat_args,
                 processes=int(pynbody.config['number_of_threads'])) :
    """

    Run a function in parallel using the python multiprocessing module.

    **Input**

    *func* : the function you want to run in parallel; there are some
             restrictions on this, see Usage below

    *single_args* : a list of arguments; only one argument from the
                    list gets passed to a single execution instance of
                    func

    *repeat_args* : a list of arguments; all of the arguments in the
                    list get passed to each execution of func

    **Optional Keywords**

    *processes* : the number of processes to spawn; default is
                  pynbody.config['number of threads']. Set to 1 for
                  testing and using the serial version of map.

    **Usage**

    Note that the function must accept only a single argument as a
    list and then expand that argument into individual inputs.

    For example:

    def f(a) :
       x,y,z = a
       return x*y*z


    Also note that the function must have a try/except clause to look
    for a KeyboardInterrupt, otherwise it's impossible to stop
    execution with ctrl+c once the code is running in the Pool. To
    facilitate this, you can use the interruptible decorator.

    To use the example from above:

    from parallel_util import interruptible

    @interruptible
    def f(a) :
        x,y,z = a
        return x*y*z

    'single_args' can be thought of as the argument you are
    parallelizing over -- for example, if you are running the same
    code over a number of different files, then 'single_args' might be
    the list of filenames.

    Similarly, 'repeat_args' are the arguments that might modify the
    behavior of 'func', but are the same each time you run the code.

    """


    from multiprocessing import Pool
    import itertools

    args = []

    if len(repeat_args) > 0:
        for arg in repeat_args:
            args.append(itertools.repeat(arg))
        all_args = zip(single_args, *args)
    else :
        all_args = single_args

    if processes==1 :
        res = list(map(func, all_args))

    else :
        pool = Pool(processes=processes)
        try :
            res = pool.map(func, all_args)
            pool.close()

        except KeyboardInterrupt :
            pool.terminate()

        finally:
            pool.join()

    return res

class KeyboardInterruptError(Exception): pass

def interruptible(func) :
    @wraps(func)
    def newfunc(*args, **kwargs):
        try :
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise KeyboardInterruptError()
    return newfunc


# from IPython.parallel import Client

# def get_particle_ids(rank,ncpu,gtot,dtot,stot):
#     ng,nd,ns = [gtot/ncpu,dtot/ncpu,stot/ncpu]
#     g_start = ng*rank
#     d_start = nd*rank+gtot
#     s_start = ns*rank+gtot+dtot

#     if rank == ncpu-1 : g_end, d_end, s_end = [gtot-1,dtot-1+gtot,stot-1+gtot+dtot]
#     else: g_end,d_end,s_end = [g_start+ng,d_start+nd,s_start+ns]

#     print(g_start, g_end, d_start, d_end, s_start, s_end)

#     g_ind = np.arange(g_start,g_end)
#     d_ind = np.arange(d_start,d_end)
#     s_ind = np.arange(s_start,s_end)

#     return np.append(np.append(g_ind,d_ind),s_ind)

# class ParallelTipsySnap(pynbody.tipsy.TipsySnap) :
#     def __init__(self, filename, **kwargs) :
#         rc = Client()
#         dview = rc[:]
#         nengines = len(rc)
#         dview.execute('import pynbody')

#         self.rc,self.dview,self.nengines = [rc,dview,nengines]

#         super(ParallelTipsySnap,self).__init__(filename,**kwargs)

#         super(ParallelTipsySnap,self).__init__(filename,**kwargs)

#         # set up particle slices

#         dview.scatter('rank',rc.ids,flatten=True)
#         dview.push({'get_particle_ids':get_particle_ids,
#                     'ncpu': self.nengines,
#                     'gtot': len(self.g),
#                     'dtot': len(self.d),
#                     'stot': len(self.s),
#                     'filename':filename})

#         dview.execute('particle_ids = get_particle_ids(rank,ncpu,gtot,dtot,stot)')
#         dview.execute('s = pynbody.load(filename,take=particle_ids)',block=True)


#     def __getitem__(self,i) :
#         if isinstance(i,str) and not self.lazy_off:
#             self.dview.execute("arr = s['%s']; units = s['%s'].units"%(i,i))
#             units = self.rc[0].pull('units').get()
#             res = self.dview.gather('arr',block=True).view(pynbody.array.SimArray)
#             res.units = units
#             res._name = i
#             res.sim = self
#             return res

#         else :
#             return super(ParallelTipsySnap,self).__getitem__(i)


# class ParallelSimArray(pynbody.array.SimArray) :
#     def sum(self, *args, **kwargs) :
#         # create individual sums on each engine
#         self.sim.dview.push({'args':args,'kwargs':kwargs})
#         self.sim.dview.execute("res = np.ndarray.sum(s['%s'],*args,**kwargs)"%self.name)
#         x = self.sim.dview.gather('res',block=True)
#         if hasattr(x, 'units') and self.units is not None :
#             x.units = self.units
#         if hasattr(x, 'sim') and self.sim is not None :
#             x.sim = self.sim
#         return x
