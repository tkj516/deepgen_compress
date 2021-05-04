clearvars
addpath mex

%% Setup
M = 512;
N = 1024*2;
t = 3;
q = 2;
seed = 123;

%% Test generate
H = full(ldpc_generate(M, N, t, q, seed));

if count(py.sys.path,pwd) == 0
  insert(py.sys.path, int32(0), pwd);
end
pyldpc_generate = py.importlib.import_module('pyldpc_generate');
py.reload(pyldpc_generate);

pyarr = py.pyldpc_generate.generate(int32(M), ...
                                    int32(N), ...
                                    t, ...
                                    int32(q), ...
                                    int32(seed)).toarray();
frompy = numpyToMat(pyarr);

assert(all(H(:) == frompy(:)))

%% test h2g
H = sparse(H);
[HH G] = ldpc_h2g(H);

pylist = py.pyldpc_generate.generateGH(int32(M), ...
                                       int32(N), ...
                                       t, ...
                                       int32(q), ...
                                       int32(seed));

% numpyToMat will convert uint8 to double. Don't worry though, the Numpy arrays
% are still uint8.
pyH = numpyToMat(pylist{1}.toarray());
pyG = numpyToMat(pylist{2});

assert(all(pyH(:) == HH(:)))
assert(all(pyG(:) == G(:)))

%% This should be the LAST thing
disp('Tests passed!')

