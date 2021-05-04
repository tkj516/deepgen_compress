# Simple Cython example

```
$ python setup.py build_ext --inplace
$ rm demo.c # this is the intermediary C file
$ python -c "import demo; print(demo.foo(2.2)); import numpy as np; x=np.arange(10).astype(np.float32); print(x); demo.scale(x,120.75); print(x)"
# ...
4.40000009537
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
[    0.     120.75   241.5    362.25   483.     603.75   724.5    845.25
   966.    1086.75]
```

# LDPC generator
```
$ python setup.py build_ext --inplace
```
Then, in Python:
```python
import pyldpc_generate
H = pyldpc_generate.generate(512,1024,3.0,2,123)
```
`H` is the sparse array of interest.

## Test
The following test requires MATLAB with `py` Python bridge and a C compiler that MEX understands.
```
$ python setup.py build_ext --inplace
$ rm demo.c ldpc_generate.c
$ cd mex && matlab -r "mex -largeArrayDims ldpc_generate.c; quit"
# make sure you quit matlab
$ cd ..
$ matlab -r "test_ldpc_generate; quit"
# ...
Tests passed!
```

If you really want to exercise the C library from a C `main`,
```
gcc -o gen.o ldpc_generate1.c gen_test.c && gen.o > gen.py
```
Confirm that this generates the same CSC sparse matrix as the Cython module wrapping the same library.
