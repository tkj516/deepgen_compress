import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
from spn_code_decoding.markov_test.utils import *
from spn_code_decoding.markov_test.markov_source import *

def test_gray_to_bin():

    x = 255
    graycoded = bin_to_gray(x, bits=8)

    expected_answer = '00000001'

    assert graycoded == expected_answer

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

if __name__ == "__main__":

    test_gray_to_bin()
    test_bin_to_gray()
    test_convert_to_graycode()