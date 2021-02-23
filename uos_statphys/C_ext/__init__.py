import platform, os, ctypes, pkg_resources
import subprocess
import numpy as np

__all__ = ["from_C_source",'set_compiler']

#global settings
FLAG = ["-shared"]
if platform.system() == 'Windows':
    CC = "clang"
    CPC = "clang++"
    outext = "dll"    
    
else:
    FLAG.append("-fPIC")
    if platform.system() == 'Linux':
        CC = "gcc"
        CPC = "g++"
        outext = "so"
    else:
        CC = "clang"
        CPC = "clang++"
        FLAG.append("-dynamic")
        outext = "dylib"

def set_compiler(c = None, cpp = None):
    global CC, CPC, FLAG
    if c:
        CC = c
    if cpp:
        CPC = cpp

class C_functions:
    def __init__(self, cdll_path, headers, doc_dict= None):
        self._path = cdll_path
        self._cdll = ctypes.CDLL(cdll_path)
        self._handle = self._cdll._handle
        interface_from_header(self._cdll, headers, c_to_py)
        types = cdll_types(headers)
        for fname in types:
            vars(self)[fname] = function_make(self._cdll, fname, *types[fname])
            if doc_dict is not None:
                vars(self)[fname].__doc__ = doc_dict[fname]

    def __getattr__(self, name: str):
        getattr(self._cdll, name)
    
    def __del__(self):
        libname = os.path.splitext(self._path)[0]
        if platform.system() == 'Windows':
            
            #ctypes.windll.FreeLibrary(self._handle)
            del self._cdll
        else:
            ctypes.cdll.LoadLibrary('libdl.so').dlclose(self._cdll._handle)
        os.remove(self._path)
        
        


# c and python interface.
c_to_py ={
    'int'       : ctypes.c_int,
    'long'       : ctypes.c_long,
    'float'     : ctypes.c_float,
    'float&'    : ctypes.c_float,
    'double'    : ctypes.c_double,
    'bool'      : ctypes.c_bool,

    'float*'    : np.ctypeslib.ndpointer(dtype=np.float32),
    'char*'     : np.ctypeslib.ndpointer(dtype=np.int8),
    'int*'      : np.ctypeslib.ndpointer(dtype=np.int32),
    'long*'      : np.ctypeslib.ndpointer(dtype=np.int64),
    'double*'   : np.ctypeslib.ndpointer(dtype=np.float64),

    'int**'     : np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
    'void'      : None
}


def c_import(c_ext):
    """Load library from C_extensions.

    Args:
        c_ext (str): path of target library.

    Raises:
        OSError: if current OS is not support "CDLL".

    Returns:
        `CDLL`: CDLL object by `ctypes`.
    """
    _cdll = ctypes.CDLL(pkg_resources.resource_filename("uos_statphys",os.path.join("lib",f"{c_ext}.{outext}")))
    return _cdll


def cdll_types(header):
    ans = {}
    for line in header:
        if line.lstrip():
            if len(line.split("("))<2: continue
            func_name = line.split("(")[0].split()[-2:]
            #print(func_name)
            restype     = (func_name[0])
            name        = func_name[1]
            func_args   = line.split("(")[1].split(")")[0].split(",")
            argtypes    = []
            argnames    = []
            for func_arg in func_args:
                args = func_arg.split()
                if args[1][0] == "*" or args[1][0] == "&":
                    argtypes.append(args[0]+args[1][0])
                    argnames.append(args[1][1:])
                else:
                    argtypes.append(args[0])
                    argnames.append(args[1])

            ans[name] = (argtypes, argnames, restype)
    return ans

def cdefs_analyzer(line):
    if line.lstrip():
        if len(line.split("("))<2: 
            return (None, None)
        func_name = line.split("(")[-2].split()[-2:]
        #print(func_name)
        restype     = (func_name[0])
        name        = func_name[1]
        func_args   = line.split("(")[-1].split(")")[0].split(",")
        argtypes    = []
        argnames    = []
        for func_arg in func_args:
            args = func_arg.split()
            if len(args) == 1:
                argtypes.append(args[0])
                continue
            if args[1][0] == "*" or args[1][0] == "&":
                argtypes.append(args[0]+args[1][0])
                argnames.append(args[1][1:])
            else:
                argtypes.append(args[0])
                argnames.append(args[1])

        return name, (argtypes, argnames, restype)
    else:
        return (None, None)

def c_interface(_cdll, func_name, argtypes, restype = None):
    func = eval(f'_cdll.{func_name}')
    func.argtypes = argtypes
    if restype is not None:
        func.restype = restype

def interface_from_header(cdll, header, py_dict, include = None, exclude = None):
    cdll.header = header
    c_funcs = cdll_types(header)
    if exclude is None:
        exclude = []
    for f_name in c_funcs:
        if f_name in exclude: continue
        if include is not None and not f_name in include: continue
        args, anames,  res = c_funcs[f_name]
        args = [py_dict[arg] for arg in args]
        res  = py_dict[res]
        c_interface(cdll, f_name, args, res)

def function_make(cdll, fname, argtypes, args, restype):
    argstring =[]
    for a,t in zip(args, argtypes):
        if t[-2:] == '**':
            argstring.append(f"{a}: np.ndarray")
        elif t[-1] == '*':
            argstring.append(f"{a}: np.ndarray")
        elif t[-1] == '&':
            argstring.append(f"{a}: {t[:-1]}")
        elif t == "double":
            argstring.append(f"{a}: float")
        else:
            argstring.append(f"{a}:{t}")
    argstring = ','.join(argstring)
    restype = None if restype =='void' else restype
    arg = ",".join(args)
    convert = ""
    for name, types in zip(args, argtypes):
        if hasattr(c_to_py[types],"_dtype_"):
            convert+=f"\t\tif not isinstance({name}, np.ndarray):\n"
            convert+=f"\t\t\t{name}=np.array({name}, dtype=np.{c_to_py[types]._dtype_})\n"
            # TODO : dtype conversion
    
    pdef = f"def functions(cdll):\n"
    pdef += f"\tdef {fname}({argstring})-> {restype}:\n"
    if convert:
        pdef += convert
    #pdef += f"\t\tprint(c, a, b, n)\n"
    
    if restype:
        pdef += f"\t\treturn cdll.{fname}({arg})\n"
    else:
        pdef += f"\t\tcdll.{fname}({arg})\n"
    pdef += f"\treturn {fname}"
    #print(pdef)
    exec(pdef)
    return locals()['functions'](cdll)


def function_def_check(file):
    doc = ""
    line = file.readline()
    funcdef = ""
    ##docstring processing
    if line[:3] == "//!":
        while line[:3] == "//!":
            doc += line[3:]
            line = file.readline()
    while not "(" in line:
        line = file.readline()
    ##function def processing
    if "(" in line:
        while not ")" in line:
            funcdef+= line[:-1]
            line = file.readline()
        funcdef+= line[:-1]
    return doc, funcdef


def precompile(filename, tempname, encoding = 'utf8', debug = False):
    cpp = not (os.path.splitext(filename)[1].lower() == "c")
    with  open(filename, encoding=encoding) as cppsource:
        header = []
        functions = []
        docs = []
        header_line = -1
        defs = False
        defse = False
        
        for i, line in enumerate(cppsource):
            if line[:16] == "//@extern_python":
                header_line = i
            if line[:13] == "//@python_def":
                doc, funcdef  = function_def_check(cppsource)
                docs.append(doc)
                functions.append(")".join(funcdef.split(")")[:-1])+");\r\n")
            if line[:13] == "//@python_var":
                functions.append(cppsource.readline())
        if header_line == -1:
            raise SyntaxError("Cannot find '//@extern_python' from source.")

    func_dict = {}
    doc_dict = {}
    if debug:
        print("C functions found from source : ")
    for docstring, func in zip(docs, functions):
        name, types = cdefs_analyzer(func)
        func_dict[name] = types
        doc_dict[name] = docstring
        if debug:
            print(f"\t{name}")
        

    #make new C source for python 
    with open(tempname,"w") as cpp_source:
        with open(filename, encoding=encoding) as cppsource:
            for i,line in enumerate(cppsource):
                if i<header_line:
                    func_name, _ = cdefs_analyzer(line)
                    if func_name in func_dict:
                        continue
                    else:
                        cpp_source.write(line)
                elif i == header_line:
                    if cpp:
                        cpp_source.write('extern "C"{\r\n')
                    for cdefs in functions:
                        if platform.system() == 'Windows':
                            cpp_source.write("__declspec(dllexport) "+cdefs)
                        else:
                            cpp_source.write(cdefs)
                    if cpp:
                        cpp_source.write('};\r\n')
                        
                    cpp_source.write(line)
                else:
                    cpp_source.write(line)
    return functions, doc_dict, cpp

def compile(filename, outputname, *flags, compiler=None, cpp = False, message= True):
    """compile give C(C++) code with option in flags.
    Parameters
    -----------
    filename : str
        path of target file.
    outputname : str
        path of output file
    flags : 
        the options for compiler
    compiler :  str
        compile with specified compiler
    cpp : bool
        the language of source code
    message : bool
        if True, the compile command line will be printed.
    """
    if compiler is not None:
        c_line = f'{compiler} -o {outputname} {filename} '
    else:
        c_line = f'{CPC} -o {outputname} {filename} ' if cpp else f'{CC} -o {outputname} {filename} '
    c_line += " ".join(flags)

    #compile
    if message:
        print(f"Compile : {c_line}")

    compile_result = subprocess.run(c_line.split(), capture_output=True)
    if  compile_result.returncode != 0:
        try:
            print(compile_result.stderr.decode())
        except UnicodeDecodeError:
            print(compile_result.stderr.decode('euckr'))
        raise SyntaxError("Compilation failed. see the compiler error output for details.")
        
    
    

def compile_option_parser(filename, encoding = "utf8"):
    """from source_code, reading out the compilation options and compiler."""
    compiler, flags = None, []
    output = False
    read = False
    with open(filename,'r', encoding=encoding) as f:
        for line in f:
            if line[:3] == "//$":
                if not read:
                    read = True
                else:
                    print("Compilation parsing failed. //$ line will be ignored.")
                    return None, []
                cline = line[3:].rstrip().split()
                for i, term in enumerate(cline):
                    if i == 0:
                        compiler = term
                        continue
                    
                    elif output:
                        output = False
                        continue
                    elif term == "-o":
                        output = True
                        continue
                    elif filename in term:
                        continue
                    else:
                        flags.append(term)
            


    if compiler is not None:
        compile(filename, "./__temp__test__source.test", *flags, compiler=compiler, message=False)
        os.remove("./__temp__test__source.test")

    return compiler, flags





def from_C_source(filename, *flags, debug = False, encoding = 'utf8', output= None):
    tempname = "__temp__"+os.path.basename(filename)
    libname = os.path.splitext(os.path.basename(filename))[0]

    functions, doc_dict, cpp = precompile(filename, tempname, encoding, debug)
    
    compiler, flag = compile_option_parser(filename, encoding)
    if compiler is None:
        c_opt = [i for i in flags]
    else:
        c_opt = [i for i in flag]
        for opt in flag:
            if not opt in c_opt:
                c_opt.append(opt)
    for opt in FLAG:
        if not opt in c_opt:
            c_opt.append(opt)

    compile(tempname, f"lib{libname}.{outext}", *c_opt, compiler=compiler, cpp= cpp)    
    if not debug:
        os.remove(tempname)

    #loadLibrary
    if doc_dict:
        dll = C_functions(f"./lib{libname}.{outext}", functions, doc_dict)
    else:
        dll = C_functions(f"./lib{libname}.{outext}", functions)
    dll.source = ""
    with open(filename, encoding=encoding) as f:
        for line in f:
            dll.source += line
        
    return dll
    


def internal_Library(module_name):
    lib_path = pkg_resources.resource_filename("uos_statphys", "lib")
    src_path = pkg_resources.resource_filename("uos_statphys", os.path.join("lib",'src'))
    try:
        _cdll = ctypes.CDLL(pkg_resources.resource_filename("uos_statphys",os.path.join("lib",f"{module_name}.{outext}")))
        return _cdll
    except:
        for srcs in os.listdir(src_path):
            if os.path.splitext(os.path.basename(srcs))[0] == module_name:  # same distribution
                _,_,cpp = precompile(os.path.join(src_path, srcs), f"__temp__{srcs}")
                compile(f"__temp__{srcs}", os.path.join(lib_path,module_name)+f".{outext}", "-O2", "-std=c++14",
                f"-I{src_path}", cpp = cpp)
                os.remove(f"__temp__{srcs}")
                return ctypes.CDLL(pkg_resources.resource_filename("uos_statphys",os.path.join("lib",f"{module_name}.{outext}")))
        raise ModuleNotFoundError()
    
        
    

    
