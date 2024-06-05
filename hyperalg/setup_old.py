import os
import sys
import subprocess
from sys import platform
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from numpy.distutils.core import Extension
import argparse
import numpy as np

## Numpy header files
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = os.path.join(numpy_lib, 'core/include')
#
# Make the git revision visible.  Most of this is copied from scipy
#
# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename='version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()

    a = open(filename, 'w+')
    try:
        a.write(cnt % dict(git_revision=GIT_REVISION))
    finally:
        a.close()


write_version_py()

cython_flags = ["-I"] + ["-v"] + ["-X embedsignature=True"]


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'cythonize.py'),
                         'hyperalg'] + cython_flags,
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


if os.name == 'posix':
    if platform == 'linux':
        os.environ["CC"] = "g++"
        os.environ["CXX"] = "g++"
    elif platform == 'darwin':
        os.environ["CC"] = "clang++"
        os.environ["CXX"] = "clang++"
    else:
        raise RuntimeError

generate_cython()


class ModuleList:
    def __init__(self, **kwargs):
        self.module_list = []
        self.kwargs = kwargs

    def add_module(self, filename):
        modname = filename.replace("/", ".")
        modname, ext = os.path.splitext(modname)
        self.module_list.append(Extension(modname, [filename], **self.kwargs))

# extract -c flag to set compiler
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-j", type=int, default=1)
parser.add_argument("-c", "--compiler", type=str, default=None)
parser.add_argument("--omp", action='store_true', default=False)
jargs, remaining_args = parser.parse_known_args(sys.argv)

if "--omp" in sys.argv:
    #from https://github.com/scipy/scipy/blob/master/setup.py#L587
    sys.argv.remove('--omp')

setup(name='hyperalg',
      version='0.1',
      author='Stefano Martiniani, Aaron Shih',
      description="Collection of algorithms to generate hyperuniform structures",
      install_requires=["numpy", "cython"],
      packages=["hyperalg"],
      )

#
# build the c++ files
#

include_sources_hyperalg = ["source/" + f
                             for f in os.listdir("source")
                             if f.endswith(".cpp") or f.endswith(".c")]

include_sources_finufft = ["../finufft/src/" + f
                             for f in os.listdir("../finufft/src/")
                             if f.endswith(".cpp") or f.endswith(".c")]
include_dirs = [numpy_include, "../finufft/include", "../finufft", "source"]

depends_hyperalg = [os.path.join("source/hyperalg", f) for f in os.listdir("source/hyperalg/")
                        if f.endswith(".cpp") or f.endswith(".c") or f.endswith(".h") or f.endswith(".hpp")]
depends_finufft = [os.path.join("../finufft/include/", f) for f in os.listdir("../finufft/include/")
                        if f.endswith(".cpp") or f.endswith(".c") or f.endswith(".h") or f.endswith(".hpp")]

# record c compiler choice. use unix (gcc) by default
# Add it back into remaining_args so distutils can see it also
idcompiler = None
if not jargs.compiler or jargs.compiler in ("unix", "gnu", "gcc"):
    idcompiler = "unix"
elif jargs.compiler in ("intel", "icc", "icpc"):
    idcompiler = "intel"

extra_compile_args = ["-std=c++17", "-Wall", "-Wextra", "-pedantic", "-O3", "-fPIC"]
extra_link_args = ["-std=c++17"]
if idcompiler.lower() == 'intel':
    extra_compile_args += ['-axCORE-AVX2', '-ipo', '-ip', '-unroll',
                           '-qopt-report-stdout']
    if jargs.omp is True:
        extra_compile_args += ['-qopenmp', '-qopt-report-phase=openmp']
else:
    extra_compile_args += ['-march=native', '-flto']
    if jargs.omp is True:
        extra_compile_args += ['-fopenmp']
        #extra_compile_args += ['-lgomp']


print('compiler options: ', extra_compile_args)

include_sources_all = include_sources_hyperalg

depends_all = depends_hyperalg

print('dependences: ', depends_all)

cxx_modules = [
    Extension("hyperalg.distances._get_distance_cpp",
                  ["hyperalg/distances/_get_distance_cpp.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=extra_link_args,
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.distances._put_in_box_cpp",
                  ["hyperalg/distances/_put_in_box_cpp.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.distances._check_overlap",
                  ["hyperalg/distances/_check_overlap.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.collective_coordinates",
                  ["hyperalg/potentials/collective_coordinates.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.hertzian_potential",
                  ["hyperalg/potentials/hertzian_potential.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.inversepower_potential",
                  ["hyperalg/potentials/inversepower_potential.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.wca_potential",
                  ["hyperalg/potentials/wca_potential.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.explicit_cc_potential",
                  ["hyperalg/potentials/explicit_cc_potential.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.potentials.explicit_sk_potential",
                  ["hyperalg/potentials/explicit_sk_potential.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  ),
    Extension("hyperalg.utils.c_utils",
                  ["hyperalg/utils/c_utils.cxx"] + include_sources_all,
                  include_dirs=include_dirs,
                  extra_compile_args=extra_compile_args,
                  libraries=['m'],
                  extra_link_args=["-std=c++17"],
                  language="c++", depends=depends_all,
                  )
]

setup(
    ext_modules=cxx_modules,
)
