import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
from spn_code_decoding.markov_test.utils import *
from spn_code_decoding.markov_test.markov_source import *
from scipy.io import loadmat

def test_gray_to_bin():

    x = 255
    graycoded = bin_to_gray(x, bits=8)

    expected_answer = 128
    expected_answer_str = '10000000'

    assert graycoded[0] == expected_answer
    assert graycoded[1] == expected_answer_str

def test_bin_to_gray():

    x = 128
    out = gray_to_bin(x)

    expected_answer = 255

    assert out == expected_answer

def test_convert_to_graycode():

    x = np.array(range(0, 256, 5)).reshape(1, -1)
    graycoded = convert_to_graycode(x, bits=8)
    
    expected_answer = np.array(
        [
            0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,
            0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,
            0,1,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,
            1,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,
            1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,
            1,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,
            0,1,1,1,1,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,
            0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,1,
            1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1
        ]
    ).reshape(-1, 1)

    assert np.array_equal(graycoded, expected_answer)

def test_msg_graycode_to_int():

    M_from_code = torch.tensor(loadmat("M_from_code.mat")["M_from_code"])
    M_to_grid = loadmat("M_to_grid.mat")["M_to_grid"]

    out = msg_graycode_to_int(M_from_code, 1, 1000, bits=8).cpu().numpy()

    assert np.sum(out - M_to_grid) <= 1e-6

def test_msg_int_to_graycode():

    M_from_grid = torch.tensor(loadmat("M_from_grid.mat")["M_from_grid"])
    M_to_code = loadmat("M_to_code.mat")["M_to_code"]

    out = msg_int_to_graycode(M_from_grid).cpu().numpy()

    assert np.sum(out - M_to_code) <= 1e-5
    

if __name__ == "__main__":

    test_gray_to_bin()
    test_bin_to_gray()
    test_convert_to_graycode()
    test_msg_graycode_to_int()
    test_msg_int_to_graycode()