import cv2
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn as nn


def get_padding(kernel):

    f = kernel.shape[0]
    p = int(np.floor(f - 1) / 2)
    return (p, p)


def get_weights(kernels):

    out_channels = len(kernels)
    in_channels = 1
    row, col = kernels[0].shape
    weights = np.empty([out_channels, in_channels, row, col], dtype=np.float)
    for i, k in enumerate(kernels):
        weights[i][0] = k
    weights_t = torch.tensor(weights, dtype=torch.float)
    return nn.Parameter(weights_t)


def conv(img, kernels):

    img_t = torch.tensor(img.reshape([1, 1, *img.shape]), dtype=torch.float)
    padding = get_padding(kernels[0])
    conv_fn = nn.Conv2d(1, len(kernels), 3, padding=padding)
    orig_shape = conv_fn.weight.size()
    weights = get_weights(kernels)
    conv_fn.weight = weights
    new_shape = conv_fn.weight.size()
    assert orig_shape == new_shape
    conv_img = conv_fn(img_t)
    return conv_img


def edge_detect(img):

    kernels = []
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
    kernels.append(k)
    kernels.append(k.T)
    edge_img = conv(img, kernels)
    edge_numpy = edge_img.detach().numpy().astype(np.uint8)
    edge_images = edge_numpy.reshape([len(kernels), *img.shape])
    output = np.zeros(img.shape)
    for out in edge_images:
        output += out
    return output.astype(np.uint8)


def cv_sobel(img):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    out = sobelx + sobely
    return out.astype(np.uint8)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--image", help="Input image path")
    args = parser.parse_args()
    img = cv2.imread(args.image, 0)
    img_edge_cv2 = cv_sobel(img)
    img_edge_torch = edge_detect(img)
    cv2.imshow("pytorch", img_edge_torch)
    cv2.imshow("cv2", img_edge_cv2)
    cv2.waitKey()
