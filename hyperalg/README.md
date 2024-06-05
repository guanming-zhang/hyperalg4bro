# Hyperuniform-Generating-Algorithms

Dependencies:

Conda install:
numpy
scipy
matplotlib
dill
tqdm
hickle
numba
statsmodels
daiquiri
cython
future


pip install
jscatter
finufft


Optional:
pyfftw
pynfft

To build the project: 
python setup.py build_ext -i

#####some notes#####
*This project is based on Aaron an Stefano's hyperalg project.
*This project is for bro or SGD for particle systems
*The include path and the linked_library are hardcoded(line12 and line40 in CMakeLists.txt.in), need to adpat them to your own finufft library locations. 

