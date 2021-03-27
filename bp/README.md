# Belief Propagation Toolbox

In this folder you can find an implementation of source-code graph decoding using belief propagation.  There are three versions of the code here for reference:

1. Python reimplementation of Ying-zong's MATLAB belief propagation code using Numpy.
2. An implementation of the MATLAB code using PyTorch instead of Numpy.
3. A fast parallelized and vectorized belief propagation code using PyTorch.

Note that method 2 is extremely slow since PyTorch is not optimized for serial operations.  Method 1 is slightly slower than its MATLAB counterpart while Method 3 is amazingly fast even while running on just the CPU.

## Optimizations Introduced (Method 3)

Sequential code runs much slower in Python in comparison to MATLAB.  This is because MATLAB uses JIT in the background which faciliates faster sequential processing.  To improve the run time of the code I optimized it by vectorizing all operations in PyTorch for parallel execution on the CPU or a powerful GPU.  Here are the key modifications:

1. The Gibb's sampling procedure was parallelized by adopting a chromatic sampling scheme.  For an Ising model we can use two colors to color the graph.  Thus we can parallelize the sampling procedure by dividing the nodes into two sets and sampling both set in parallel.
2. The computational bottleneck is the code graph decoding.  In order to speed up code graph belief propagation all message calculations were parallelized.  This can be done with the same precision as MATLAB by using PyTorch.  They key to this implementation is to understand logical indexing by converting if-else conditions to logical indexing operations over the entire message tensor.
3. Minimizing data storage and transmission is key.  An advantage of Python over MATLAB is the ability to reuse and store variables efficiently using classes.  Memoization is also very helpful to prevent redundant calculations.
4. Converting the three for loops to vectorized implementations results in nearly a 40x speed improvement in some cases.

## Running the code

To run belief propagation on a Gibb's sampled image of size 128x128, use the following command:

python test_source_code_bp_torch.py 

To run it on a pre-stored image, use the following command:

python test_source_code_bp_torch.py --load_image

The same image when loaded in MATLAB requires a decoding time of 87s while the PyTorch code was able to decode in just 2s

For the numpy version of the code look under the numpy_version folder and for the iterative PyTorch code look under the torch_iterative folder.