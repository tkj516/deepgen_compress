% Taken from answer by Jim Hokanson
% http://www.mathworks.com/matlabcentral/answers/157347-convert-python-numpy-array-to-double
%   if a simple list would be like this:
% means = cellfun( @double, cell(ms))


function data = numpyToMat(x)

    data_size = cell2mat(cell(x.shape));
    % if empty array
    if data_size == 0
        data = [];
        return
    end

    data_row = double(py.array.array('d', py.numpy.nditer(x, pyargs('order', 'F'))));  % Add order='F' to get data in column-major order (as in Fortran 'F' and Matlab
    if length(data_size) > 1
        data = reshape(data_row, data_size);  % No need for transpose, since we're retrieving the data in column major order
    else
        data = data_row;
    end
