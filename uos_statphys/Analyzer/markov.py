from re import error
import re
from uos_statphys.isingModel import Observable
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
import tqdm

__all__ = []

class Container(object):
    pass

class Observable:
    pass

class SingleDataSet:
    """A class for analysis of monte Carlo simulation result."""
    def __init__(self, **parameters):
        self.parameters = parameters
        self.__analyzed = False
        self.ensemble = None
        self.simulation_time = None
    
    @classmethod
    def from_RAW(cls, files, obs_names, delimiter = ",", **parameters):
        """
        we assume that each file is a single ensemble of simulation and each ensemble has a same simulation time.
        """
        pass

    @classmethod
    def from_npy(cls, arrays, obs_names, ensemble_axis = None, **parameters):
        """
        we assume that each ensemble has a same simulation time.
        """
        pass

    def set_parameters(self, **parameters):
        for p in parameters:
            self.parameters[p] = parameters[p]

    def set_order_parameter(self, var_name):
        self.order_parameter = var_name


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
        
        
                
        self.__analyzed = True

    def save(self, file, format="npy"):
        pass
        
    
        

class SingleAnalyzer:
    """A class for single control parameter analysis of monte Carlo simulation result."""
    def __init__(self):
        self.__analyzed = False
    
    def set_parameters(self, **parameters):
        var = vars(self)
        for p in parameters:
            var[p] = parameters[p]

    def set_order_parameter(self, var_name):
        self.order_parameter = var_name

    def set_control_parameter(self, var_name):
        self.control_parameter = var_name

    def append(self, data, **parameters):
        pass
    
    def analyze(self):
        pass

    @classmethod
    def from_RAW(cls, files, var_names, delimiter = ",", **parameters):
        """
        we assume that each file is a single ensemble of simulation and each ensemble has a same simulation time.
        """
        pass

    @classmethod
    def from_npy(cls, arrays, var_names, parameter_axis, ensemble_axis = None, **parameters):
        """
        we assume that each ensemble has a same simulation time.
        """
        pass
                
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

    def __getattr__(self, value):
        return self.parameters[value]

    def new_variable_with(self, *arg): # define new variable as property
        param = map(arg, self.parameters)
        def define_new(func): #decorator
            def variable():
                doc = """Variable difined by user."""
                def fget():
                    return func()
                def fset():
                    raise AttributeError("can't set attribute")
                def fdel():
                    return 
                return locals()
            vars(self)[func.__name__] = property(**locals())
            return 
        vars(self)[func.__name__] = define_new
        return define_new

    def new_observable_with(self, *arg): # define new observable
        
        def define_new(func): #decorator
            return 
        
        return define_new()    
    
    def plot():
        pass
    
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

class MultiAnalyzer(SingleAnalyzer):
    """A class for single control parameter analysis of monte Carlo simulation result."""
    def __init__(self):
        self.__analyzed = False
    
    def set_parameters(self, **parameters):
        var = vars(self)
        for p in parameters:
            var[p] = parameters[p]

    def set_order_parameter(self, var_name):
        self.order_parameter = var_name

    def set_control_parameter(self, var_name):
        self.control_parameter = var_name

    def append(self, data, **parameters):
        pass
    
    def analyze(self):
        pass
    
    @classmethod
    def from_RAW(cls, files, var_names, delimiter = ",", **parameters):
        """
        we assume that each file is a single ensemble of simulation and each ensemble has a same simulation time.
        """
        pass

    @classmethod
    def from_npy(cls, arrays, var_names,parameter_axis, ensemble_axis = None, **parameters):
        """
        we assume that each ensemble has a same simulation time.
        """
        pass
                
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

    def __getattr__(self, value):
        return self.parameters[value]

    def new_variable_with(self, *arg): # define new variable as property
        param = map(arg, self.parameters)
        def define_new(func): #decorator
            def variable():
                doc = """Variable difined by user."""
                def fget():
                    return func()
                def fset():
                    raise AttributeError("can't set attribute")
                def fdel():
                    return 
                return locals()
            vars(self)[func.__name__] = property(**locals())
            return 
        vars(self)[func.__name__] = define_new
        return define_new

    def new_observable_with(self, *arg): # define new observable
        
        def define_new(func): #decorator
            return 
        
        return define_new()    
    
    def plot():
        pass
    
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