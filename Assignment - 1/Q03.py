import cv2
from skimage.util import random_noise
import numpy as np


def snpNoise(img):
    return np.array(255 * random_noise(img, mode='s&p', amount=0.1), dtype="uint8")


def gaussianNoise(img):
    return np.array(255 * random_noise(img, mode='gaussian', mean=0, var=0.01), dtype="uint8")


def speckleNoise(img):
    row, col = img.shape
    gauss = np.random.randn(row, col)
    noisy = img + img * gauss
    return noisy


def getMean(img, noise, n):
    arr = []
    for _ in range(n):
        arr.append(functions[noise](img))
    return np.mean(arr, axis=0).astype('uint8')


def getOutput(img, noise):
    for i in range(5, 35, 5):
        cv2.imwrite(
            f"Output/Q03/{noise}/lena[{i}].png", getMean(img, noise, i))


if __name__ == "__main__":
    functions = {
        "s&p": snpNoise,
        "gaussian": gaussianNoise,
        "speckle": speckleNoise
    }
    img = cv2.imread('Resources/lena.png', cv2.IMREAD_GRAYSCALE)
    for i in functions.keys():
        getOutput(img, i)
        print(f"Output images generated at Output/Q03/{i}.")
