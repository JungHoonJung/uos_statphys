__all__ = ['IsingModel','IsingMultiAnalyzer']

from .ImportManager import Import_Manager        
mm = Import_Manager()
mm.requireAs('ctypes', 'os','gc', np = 'numpy', plt = 'matplotlib.pyplot', globs = globals())
import ctypes, os, gc
import numpy as np
import matplotlib.pyplot as plt
from .C_ext import internal_Library

class IsingModelSingle:
    '''class for 2D square lattice ising model simulator (only one size).
    for now, there are two methods, metropolis and wolff algorithm.
    
    Parameters
    ------------
    L : `int`
        A size of system. total number of spin is N (L*L).
    algorithm : `str`
        specifying Monte Carlo algorithm. 'wolff' and 'metropolis' can be captured. Default is 'metropolis'
    '''
    _cdll = internal_Library('isingmonte')
    metropolis = _cdll.monteCarlo
    metropolis.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                 np.ctypeslib.ndpointer(dtype=np.int32),
                 ctypes.c_double, 
                 ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int]
    wolff = _cdll.wolff
    wolff.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                 np.ctypeslib.ndpointer(dtype=np.int32),
                 ctypes.c_double, 
                 ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int]
    
    _modules = {'np':'numpy'}
    
    
    def __init__(self, L, algorithm = 'wolff'):

        self.L = L
        self.T_range = []
        self.relax = []
        self.simulate_time = []
        self.algorithm = 'wolff'
        if not algorithm in ['wolff', 'metropolis']:
            raise ValueError("Wrong algorithm, only 'wolff' and 'metropolis' are allowed. ")
            
    def __repr__(self):
        return f"<Single ising model simulator>\nL\t\t : {self.L}\n"+f"T\t\t : {self.T_range}\n"+f"MC_steps\t : {self.simulate_time}"
        
    def __gt__(self, other):
        return self.L > other.L
    
    def __ge__(self, other):
        return self.L >= other.L

    def add_set(self, temp, relaxation, MC_step):
        ''' add simulation setting for single size simulator.
        
        Parameters
        ------------
        temp : `list`
            Temperatures which will be simulated.
        relaxation : `int`
            Relaxation iterations. Iteration number of pre-steps.
        MC_step : `int`
            MC iterations. Iteration number for simulation.
        
        '''
        if isinstance(temp, int) or isinstance(temp, float):
            self.T_range.append(temp)
            self.relax.append(relaxation)
            self.simulate_time.append(MC_step)
        else:
            temp = list(temp)
            if isinstance(relaxation, list):
                if len(temp)!=len(relaxation):
                    raise ValueError("'relaxation' must be `int` or have same length with `temp`.")
            else:
                relaxation = [int(relaxation) for i in temp]
            if isinstance(MC_step, list):
                if len(temp)!=len(MC_step):
                    raise ValueError("'MC_step' must be `int` or have same length with `temp`.")
            else:
                MC_step = [int(MC_step) for i in temp]
            self.T_range+=temp
            self.relax+=relaxation
            self.simulate_time+=MC_step
        
            
            
    
            
    def _sort(self):
        temp = np.array([self.T_range, self.relax, self.simulate_time])
        temp.T.sort(axis = 0)
        self.T_range = temp[0]
        self.relax, self.simulate_time = temp.astype(np.int32)[1:]
        self.T_range = list(self.T_range)
        self.relax = list(self.relax)
        self.simulate_time = list(self.simulate_time)
        
    def simulate(self, ensemble, thr_num = 1):
        global mm
        if mm.check('tqdm'):
            tqdm = mm.require_func(tqdm = 'tqdm', globs = globals())
        else:
            tqdm = lambda x:x
        if not self.T_range:
            raise ValueError("no target temperature. please `add_set` to add target temperature.")
        self.E, self.M = [],[]
        self.ensemble = ensemble
        if (not isinstance(ensemble, int)) or (ensemble <= 0 ):
            raise ValueError("the value of 'ensemble' is invalid.")
            
        self._sort()
        for T, rel, nsteps in tqdm(list(zip(self.T_range,self.relax, self.simulate_time))):
            energy = np.zeros([self.ensemble*nsteps],dtype = np.int32)
            mag = np.zeros([self.ensemble*nsteps],dtype = np.int32)
            vars(self.__class__)[self.algorithm](energy, mag, T, self.ensemble, self.L, rel, nsteps, thr_num)
            self.E.append(energy.reshape([self.ensemble, -1]))
            self.M.append(mag.reshape([self.ensemble, -1]))
        self.E = np.array(self.E)
        self.M = np.array(self.M)
        
    def save(self, path):
        if os.is_dir(path):
            pass
        
    
    def get_analyzer(self):
        temp = IsingSingleAnalyzer(self.L, np.array(self.T_range), self.E, self.M)
        temp.total_ensemble = self.ensemble
        temp.sim_time       = self.simulate_time
        return temp
            

class IsingModel:
    '''class for 2D square lattice ising model simulator.
    for now, there are two methods, metropolis and wolff algorithm.
    
    Parameters
    ------------
    algorithm : `str`
        specifying Monte Carlo algorithm. 'wolff' and 'metropolis' can be captured. Default is 'metropolis'
    '''
    _modules = {'np':'numpy'}
    def __init__(self, algorithm = 'wolff'):
        self.entry = []
        self.algorithm = algorithm
        
        
    @property
    def algorithm(self):
        return self.__algorithm
    @algorithm.getter
    def algorithm(self):
        return self.__algorithm
    @algorithm.setter
    def algorithm(self, value):
        if not value in ['wolff', 'metropolis']:
            raise ValueError("Wrong algorithm, only 'wolff' and 'metropolis' are allowed. ")
        for ISim in self.entry:
            ISim.algorithm = value
        self.__algorithm  = value
        
    def __repr__(self):
        return "<Ising Model Simulator>"

    def show_setting(self):
        for i in self.entry:
            print(i)

    def add_set(self, L, temp, relaxation, MC_step):
        ''' add simulation setting for single size simulator.
        
        Parameters
        ------------
        L : `int`
            size of system.
        temp : `list`
            Temperatures which will be simulated.
        relaxation : `int`
            Relaxation iterations. Iteration number of pre-steps.
        MC_step : `int`
            MC iterations. Iteration number for simulation.
        
        '''
        
        if isinstance(L, int):
            L = [L]
        
            
        for l in L:
            ch = False
            for ISim in self.entry:
                if ISim.L == l:
                    ch = True
                    ISim.add_set(temp, relaxation, MC_step)
            if not ch:
                ims = IsingModelSingle(l, self.algorithm)
                ims.add_set(temp, relaxation, MC_step)
                self.entry.append(ims)
                
    def _sort(self):
        self.entry.sort()
        for ISim in self.entry:
            ISim._sort()
    
    def simulate(self, ensemble, thr_num = 1):
        self.ensemble = ensemble
        if (not isinstance(ensemble, int)) or (ensemble <= 0 ):
            raise ValueError("the value of 'ensemble' is invalid.")
        for ISim in self.entry:
            ISim.simulate(ensemble, thr_num)
        
    def __getitem__(self, value):
        for i in self.entry:
            if i.L == value:
                return i
        raise KeyError(f'{value}')
        
    def get_analyzer(self):
        temp = IsingMultiAnalyzer.new()
        for Isim in self.entry:
            temp.append(Isim.get_analyzer())
        return temp
        

class IsingSingleAnalyzer:
    _modules = {'np':'numpy'}
    def __init__(self, L, T, E, M):
        self.L = L
        self.T = T
        self.E = E
        self.M = np.abs(M)
        
        self.total_ensemble = None
        self.sim_time       = None
        if isinstance(E, np.ndarray):
            assert E.shape == M.shape
            assert E.shape[0]==T.shape[0]
            assert M.shape[0]==T.shape[0]

            self.total_ensemble = E.shape[1]
            self.sim_time = E.shape[2]
            
        self.__analyzed = False
    
    def analyze(self, reduced = False):
        if self.__analyzed: return
        self.average  = Container()
        self.var      = Container()
        self.second   = Container()
        self.forth    = Container()
        
        #self.meaned   = Container()
        
        for key in vars(self).copy():
            if isinstance(vars(self)[key], np.ndarray) and len(vars(self)[key].shape)==3:
                vars(self.average)[key] = np.average(vars(self)[key], axis =2)
                vars(self.var)[key]     = np.var(vars(self)[key], axis =2)
                vars(self.second)[key]  = np.average(vars(self)[key].astype(np.float64)**2, axis =2)
                vars(self.forth)[key]   = np.average(vars(self)[key].astype(np.float64)**4, axis =2)
            if isinstance(vars(self)[key], list) and len(vars(self)[key][0].shape)==2:
                temp = [[],[],[],[]]
                for i in range(len(vars(self)[key])):
                    temp[0].append(np.average(vars(self)[key][i], axis =1))
                    temp[1].append(np.var(vars(self)[key][i], axis =1))
                    temp[2].append(np.average(vars(self)[key][i].astype(np.float64)**2, axis =1))
                    temp[3].append(np.average(vars(self)[key][i].astype(np.float64)**4, axis =1))
                vars(self.average)[key], vars(self.var)[key], vars(self.second)[key], vars(self.forth)[key] = np.array(temp)
        
        if reduced:
            del self.E, self.M
                
        self.__analyzed = True
                

class Container(object):
    pass

class Observable:
    def __init__(self, obj):
        self.__raw = obj
        self.__aver, self.__var = None, None

    @property
    def average(self):
        return self.__aver
    @average.getter
    def average(self):
        if self.__aver is None:
            self.__aver = np.average(obj, axis = 2)
    

class IsingSingleAnalyzer:
    def __init__(self, L, T, E, M):
        self.L = L
        self.T = T
        self.E = E
        self.M = np.abs(M)
        
        assert E.shape == M.shape
        assert E.shape[0]==T.shape[0]
        assert M.shape[0]==T.shape[0]
        
        self.total_ensemble = E.shape[1]
        self.sim_time = E.shape[2]
        self.__analyzed = False
    
    def analyze(self, reduced = False):
        if self.__analyzed: return
        self.average  = Container()
        self.var      = Container()
        self.second   = Container()
        self.forth    = Container()
        
        for key in vars(self).copy():
            if isinstance(vars(self)[key], np.ndarray) and len(vars(self)[key].shape)==3:
                vars(self.average)[key] = np.average(vars(self)[key], axis =2)
                vars(self.var)[key] = np.var(vars(self)[key], axis =2)
                vars(self.second)[key] = np.average(vars(self)[key].astype(np.float64)**2, axis =2)
                vars(self.forth)[key] = np.average(vars(self)[key].astype(np.float64)**4, axis =2)
        
        if reduced:
            del self.E, self.M
                
        self.__analyzed = True
        
    def reduced_T(self, t_c):
        return (self.T - t_c)/t_c
    
    def observable(func):
        def plots(self, return_errors = False, return_argmax = False):
            if not self.__analyzed:
                self.analyze()
                
            raw = func(self) #calculation
            ret = [np.mean(raw, axis = 1)]
            if return_errors:
                ret.append(np.std(raw, axis = 1))
            if return_argmax:
                ret.append(np.argmax(raw, axis = 0))
            return ret
        return plots
    
    @observable
    def susceptibility(self):
        return self.var.M/self.L/self.L/self.T
    
    @observable
    def heat_capacity(self):
        return self.var.E/self.L/self.L/self.T/self.T
    
    @observable
    def binder_cumulant(self):
        forth = self.forth.M
        second = self.second.M
        return 1 - forth/3/second**2
        
                
                
        
class IsingMultiAnalyzer:
    def __init__(self,L,T,E,M, title = ""):
        self.entry = []
        self.L = L
        for l,t,e,m in zip(L,T,E,M):
            vars(self)[f"_{l}"] = IsingSingleAnalyzer(l,t,e,m)
            self.entry.append(vars(self)[f"_{l}"])
        self.title = title
        self.__analyzed = False
        
    def new(isa = None, title =""):
        temp = IsingMultiAnalyzer([],[],[],[],title)
        if isa is not None:
            temp.append(isa)
        return temp
        
        
    def append(self, value):
        if isinstance(value, IsingSingleAnalyzer):
            self.L.append(value.L)
            self.entry.append(value)
            vars(self)[f"_{l}"] = value
        else:
            raise ValueError
            
    
    def analyze(self):
        for isa in self.entry:
            isa.analyze()
        self.__analyzed = True
            
    @property
    def average(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.average.E, isa.average.M
            
    
    @property
    def variance(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.var.E, isa.var.M
    
    @property
    def second(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.second.E, isa.second.M
            
    @property
    def forth(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.forth.E, isa.forth.M
    
    def line_fitting(self, x, y, y_err, line_range= None, logscale = False , label = ""):
        popt , pcov = curve_fit(lambda xhat,a,b:a*xhat+b, x, y, sigma =y_err )
        perr = np.sqrt(np.diag(pcov))
        if line_range is not None:
            pred_x = np.array(line_range)
            if logscale:
                pred_x = np.power(10,pred_x)
                predict = 10**popt[1]*np.power(pred_x,popt[0])
            else:
                
                predict = popt[0]*np.array(line_range)+popt[1]
                
            if label:
                plt.plot(pred_x, predict, label = label)
            else:
                plt.plot(pred_x, predict)
        return popt, perr
        
    
    def plot_setting(self, xlim= None, ylim=None, logx = False, logy = False, xtitle="", ytitle="", title = "",legend = True):
        if title:
            plt.title(title)
        else:
            if self.title:
                plt.title(self.title)
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if legend:
            plt.legend()
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        
    
    def __iter__(self):
        for isa in self.entry:
            yield isa

    def __getitem__(self, value):
        return self.entry[value]
        
def ens_mean(x):
    return np.average(x, axis=1)
def ens_std(x):
    return np.std(x, axis=1)                
                
class Container(object):
    def show_all(self):
        return [name for name in vars(self)]

    def __getitem__(self, value):
        return vars(self)[value]
    
                
class IsingMultiAnalyzer:
    _modules = {'np':'numpy', 'plt':'matplotlib.pyplot'}
    def __init__(self,L,T,E,M, title = ""):
        self.entry = []
        self.L = L
        for l,t,e,m in zip(L,T,E,M):
            vars(self)[f"_{l}"] = IsingSingleAnalyzer(l,t,e,m)
            self.entry.append(vars(self)[f"_{l}"])
        self.title = title
        self.__analyzed = False
        
    @staticmethod
    def new(title =""):
        temp = IsingMultiAnalyzer([],[],[],[],title)
        return temp
        
        
    def append(self, value):
        if isinstance(value, IsingSingleAnalyzer):
            self.L.append(value.L)
            self.entry.append(value)
            vars(self)[f"_{value.L}"] = value
        else:
            raise ValueError
            
    
    def analyze(self):
        for isa in self.entry:
            isa.analyze()
        self.__analyzed = True
            
    @property
    def average(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.average.E, isa.average.M
            
    
    @property
    def variance(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.var.E, isa.var.M
    
    @property
    def second(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.second.E, isa.second.M
            
    @property
    def forth(self):
        if not self.__analyzed:
            self.analyze()
        for l, isa in zip(self.L,self.entry):
            yield l, isa.T, isa.forth.E, isa.forth.M
            
    
    def plot_setting(self, xlim= None, ylim=None, logx = False, logy = False, xtitle="", ytitle="", title = "",legend = True):
        if title:
            plt.title(title)
        else:
            if self.title:
                plt.title(self.title)
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if legend:
            plt.legend()
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        for isa in self.entry:
            yield isa
    
    def __getitem__(self, value):
        return self.entry[value]
        
def ens_mean(x):
    return np.average(x, axis=1)
def ens_std(x):
    return np.std(x, axis=1)
        