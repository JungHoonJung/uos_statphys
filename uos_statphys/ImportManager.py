__all__ = ['']
class Import_Manager:
    '''class for import manager for classes.'''
    def __init__(self, globs= None):
        self._modules = {} 
        self._funcs = {}
        self._alls = {}
        
    def copy(self, targets, globs):
        if isinstance(targets, dict):
            self.requireAs(globs = globs, **targets)
        else:
            self.requireAs(globs = globs, **targets._modules)
    
    def check(self, module_name):
        if module_name in self._modules:
            return True
        else:
            ch = False
            try:
                self.load(module_name)
                ch = True
            except:
                ch = False
            return ch
    
    def import_all(self, module_name):
        if module_name in self._alls:
            return self._alls[module_name]
        
        module_name_split = module_name.split(".")
        if len(module_name_split) == 1:
            if module_name in self._alls:
                return self._alls[module_name]
            else:
                TARTGET_MOD_code = f"from {module_name} import *"
                LOCAL_COPY = set(locals().keys()).copy()
                exec(TARTGET_MOD_code)
                LOCAL_COPY = set(locals().keys()).copy()-LOCAL_COPY 
                self._alls[module_name] = {}
                for ALL_ATT_name in LOCAL_COPY:
                    self._alls[module_name][ALL_ATT_name] = locals()[ALL_ATT_name]
                return self._alls[module_name]
        
        
    
    def load(self, module_name, func_name = None):
        if func_name is None:
            var = self._modules
            mns = module_name.split(".")
            if len(mns) == 1:
                if module_name in var:
                    return var[module_name]
                else:
                    code = f"import {module_name}"
                    exec(code)
                    var[module_name] = locals()[module_name]
                    return var[module_name]
            else:
                module = var.get(mns[0],False)
                if module:
                    for i in range(len(mns)-1):
                        module = vars(module).get(mns[i+1], lambda x:x)
                if module:
                    return module
                
                code = f"import {module_name}"
                exec(code)
                target = locals()[mns[0]]
                for i in range(len(mns)-1):
                    target = vars(target)[mns[i+1]]
                var[module_name] = target
                return var[module_name]
        else:
            var = self._funcs
            if func_name in var:
                return var[func_name]
            else:
                code = f"from {module_name} import {func_name}"
                exec(code)
                var[func_name] = locals()[func_name]
                return var[func_name]
        
    def require(self, *args, globs = None, namespace = None):
        '''equivalent as python `import module_name`.
        Parameters
        ------------
        globs : `dict`
            imported module will be assigned into `globs`, default is None.
        
        args
        --------
        module_name : `string`
            The exact name of module, which is used in python code.
            
        Return
        ---------
        `list` of `module`
            Python modules which are exactly imported by python. the order is same with `args`.
            If `globs` is not `None`, return is `None`.
            
        examples
        ----------
        this two codes are equivalent
        ```
        import numpy 
        import ctypes
        import os, sys
        ```
        and 
        `Import_Manager.require('numpy','ctypes','os','sys', globs= globals())`
        '''
        info = None
        if namespace is not None:
            info = vars(namespace).get('_modules', {})
        targets = {}
        if globs is not None:
            targets = globs
            
        if len(args) == 1:
            if namespace is not None:
                info[args[0]] = args[0]
            targets[args[0]] = self.load(args[0])
            return targets[args[0]]
        
        for mn in args:
            targets[mn] = self.load(mn)
            
        if namespace is not None:
            vars(namespace)['_modules'] = info
        if (globs is None):
            return [targets[mod] for mod in args]
        
            
    def requireAs(self, *args, globs = None,namespace = None, **kwargs):    
        '''equivalent as python `import module_name as assigned_name`.
        Parameters
        ------------
        globs : `dict`
            imported module will be assigned into `globs`, default is None.
        
        args
        --------
        module_name : `string`
            The exact name of module, which is used in python code.
            
        kwargs
        --------
        assign_name = module_name : `string`
            `key` is variable which you want to use, and value is the exact name of module, which is used in python code.
            
        Return
        ---------
        `dict` of `module`
            Python modules which are exactly imported by python. the order is same with `args`.
            If `globs` is not `None`, return is `None`.
            
        examples
        ----------
        Below two codes are equivalent
        ```
        import numpy as np
        import matplotlib.pyplot as plt
        import ctypes
        import os, sys
        ```
        and 
        `Import_Manager.requireAs(np = 'numpy', plt = 'matplotlib.pyplot','ctypes','os','sys', globs= globals())`
        '''
        targets = {}
        if globs is not None:
            targets = globs

        
        if args:
            self.require(*args,globs = targets, namespace = namespace)
        if namespace is not None:
            info = vars(namespace).get('_modules', {})
        for an in kwargs:
            targets[an] = self.load(kwargs[an])
            if namespace is not None:
                info[an] = kwargs[an]
        if namespace is not None:
            vars(namespace)['_modules'] = info
            
        if (globs is None):
            return targets

    def require_func(self, globs, **kwargs):
        '''equivalent as python `from module_name import func_name`.
        Parameters
        ------------
        globs : `dict`
            imported module will be assigned into `globs`, default is None.
        
            
        kwargs
        --------
        func_name = module_name : `string`
            `key` is function which you want to import, and value is the exact name of its module, which is used in python code.
            
        Return
        ---------
        `dict` of `function`
            Python modules which are exactly imported by python. the order is same with `args`.
            
        examples
        ----------
        Below two codes are equivalent
        ``from tqdm import tqdm
        from matplotlib.pyplot import plot``
        and 
        `Import_Manager.require_func(tqdm = 'tqdm', plot = 'matplotlib.pyplot')`
        '''
        targets = globs
        
        
        _all = False
        for fn in kwargs:
            if fn[:3]=='all':
                _all = True
                temp = self.import_all(kwargs[fn])
                for key in temp:
                    targets[key] = temp[key]
            else:
                targets[fn] = self.load(kwargs[fn], fn)
        if not _all and len(kwargs) == 1:
            return targets[fn]