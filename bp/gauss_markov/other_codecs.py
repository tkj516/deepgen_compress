# Taken from https://github.com/fab-jul/RC-PyTorch/blob/5064d604a6370276c2132e82342f071df3b53c6b/src/lossy/other_codecs.py#L474

import os
from PIL import Image
import numpy as np
import itertools
from contextlib import contextmanager
import subprocess
import re
import torchvision
from torchvision.datasets import CIFAR10
import json

os.environ['PATH'] += ':' + '/fs/data/tejasj/kakadu/bin'
os.environ['PATH'] += ':' + '/fs/data/tejasj/kakadu'
os.environ['LD_LIBRARY_PATH'] = ':' + '/fs/data/tejasj/kakadu/bin'
os.environ['LD_LIBRARY_PATH'] += ':' + '/fs/data/tejasj/kakadu'

print(os.environ)

KDU_COMPRESS = os.environ.get('KDU_COMPRESS', '/fs/data/tejasj/kakadu/kdu_compress')
_KDU_RE_PAT = r'Compressed bytes \(excludes codestream headers\) = .*=\s(.*)\sbpp'


def jp_compress(input_image_p, q):
    output_image_jp_p = os.path.splitext(input_image_p)[0] + '_tmp_out_jp_{}.jpg'.format(q)
    img = Image.open(input_image_p)
    img.save(output_image_jp_p, quality=q, subsampling=0)
    dim = float(np.prod(img.size))
    bpp = (8 * _jpeg_content_length(output_image_jp_p)) / dim
    return bpp, output_image_jp_p


def jp_compress_accurate(input_image_p, img, target_bpp, verbose=False):
    out_path = os.path.splitext(input_image_p)[0] + '_out_jp.jpg'
    # img = Image.open(input_image_p)
    dim = float(img.size[0] * img.size[1])
    for q in range(1, 99):
        img.save(out_path, quality=q)
        bpp = (8 * _jpeg_content_length(out_path)) / dim
        if bpp > target_bpp:
            if verbose:
                print('q={} -> {}bpp'.format(q, bpp))
            return out_path, bpp
    # raise ValueError(
    #         'Cannot achieve target bpp {} with JPEG for image {} (max {})'.format(target_bpp, input_image_p, bpp))

    return -1, -1


def _jpeg_content_length(p):
    """
    Determines the length of the content of the JPEG file stored at `p` in bytes, i.e., size of the file without the
    header. Note: Note sure if this works for all JPEGs...
    :param p: path to a JPEG file
    :return: length of content
    """
    with open(p, 'rb') as f:
        last_byte = ''
        header_end_i = None
        for i in itertools.count():
            current_byte = f.read(1)
            if current_byte == b'':
                break
            # some files somehow contain multiple FF DA sequences, don't know what that means
            if header_end_i is None and last_byte == b'\xff' and current_byte == b'\xda':
                header_end_i = i
            last_byte = current_byte
        # at this point, i is equal to the size of the file
        return i - header_end_i - 2  # minus 2 because all JPEG files end in FF D0

@contextmanager
def remove_file_after(p):
    yield p
    os.remove(p)

def convert_im_to(ext, input_image_p):
    input_image_root_p, _ = os.path.splitext(input_image_p)
    im = Image.open(input_image_p)
    input_image_ext_p = input_image_root_p + '__tmp.{}'.format(ext)
    im.save(input_image_ext_p)
    return input_image_ext_p

def jp2k_compress(input_image_p, target_bpp, no_weights=True):
    output_image_j2_p = os.path.splitext(input_image_p)[0] + '_out_jp2.jp2'
    # kdu can only work with "tif", "tiff", "bmp", "pgm", "ppm", "raw" and "rawl"
    with remove_file_after(convert_im_to('bmp', input_image_p)) as input_image_bmp_p:
        cmd = [KDU_COMPRESS,
               '-i', input_image_bmp_p, '-o', output_image_j2_p,
               '-rate', str(target_bpp), '-no_weights']
        output = subprocess.check_output(cmd).decode()
        actual_bpp = float(re.search(_KDU_RE_PAT, output).group(1))
        return output_image_j2_p, actual_bpp


def jp2k_compress_accurate(input_image_p, target_bpp, verbose=False, delta=0.005):
    for i in range(25):
        out_path, actual_bpp = jp2k_compress(input_image_p, target_bpp + i * delta)
        if actual_bpp >= target_bpp:
            if verbose:
                print('target={} -> actual={}bpp'.format(target_bpp, actual_bpp))
            return out_path, actual_bpp
    # raise ValueError(
    #         'Cannot achieve target bpp {} with JPEG2K for image {} (max {}bpp)'.format(
    #                 target_bpp, input_image_p, actual_bpp))
    return -1, -1


def get_jpeg_bpp():

    transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
            ])
    dataset = CIFAR10('../../../CIFAR10', train=False, transform=transform)

    results = []

    for i in range(10):

        s, _ = dataset[i]

        for bpp in np.arange(1, 8, 0.5):

            out_path, actual_bpp = jp_compress_accurate('temp.png', s, bpp, verbose=True)
            if out_path == -1:
                continue
            
            recon = Image.open(out_path)

            orig_norm = 2 * np.array(s) / 256 - 1
            recon_norm = 2 * np.array(recon) / 256 - 1

            mse = np.mean((orig_norm - recon_norm)**2)
            sqnr = -10 * np.log10(mse)

            print(sqnr)

            results.append((i, bpp, actual_bpp, mse, sqnr))

    with open('temp4.json', 'w') as file:
        json.dump(results, file, indent=4)

    filepath = os.path.join('demo_gray_lossy', 'cifar-10', 'results_jpeg.json')
    os.makedirs(os.path.join('demo_gray_lossy', 'cifar-10'), exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

def get_jp2k_bpp():

    transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
            ])
    dataset = CIFAR10('../../../CIFAR10', train=False, transform=transform)

    results = []

    for i in range(10):

        s, _ = dataset[i]
        # Save into a temporary file
        s.save('temp.tiff')

        for bpp in np.arange(0.01, 8, 0.01):

            out_path, actual_bpp = jp2k_compress_accurate('temp.tiff', bpp, verbose=True)
            if out_path == -1:
                continue
            
            recon = Image.open(out_path)

            orig_norm = 2 * np.array(s) / 256 - 1
            recon_norm = 2 * np.array(recon) / 256 - 1

            mse = np.mean((orig_norm - recon_norm)**2)
            sqnr = -10 * np.log10(mse)

            print(sqnr)

            results.append((i, bpp, actual_bpp, mse, sqnr))

    with open('temp5.json', 'w') as file:
        json.dump(results, file, indent=4)

    filepath = os.path.join('demo_gray_lossy', 'cifar-10', 'results_jp2k.json')
    os.makedirs(os.path.join('demo_gray_lossy', 'cifar-10'), exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    # get_jpeg_bpp()
    get_jp2k_bpp()