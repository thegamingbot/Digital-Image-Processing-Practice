import numpy as np
import math
import cv2


def GetBilinearPixel(imArr, posX, posY):
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi+1, imArr.shape[1]-1)
    modYiPlusOneLim = min(modYi+1, imArr.shape[0]-1)

    bl = imArr[modYi, modXi]
    br = imArr[modYi, modXiPlusOneLim]
    tl = imArr[modYiPlusOneLim, modXi]
    tr = imArr[modYiPlusOneLim, modXiPlusOneLim]

    b = modXf * br + (1. - modXf) * bl
    t = modXf * tr + (1. - modXf) * tl

    pxf = modYf * t + (1. - modYf) * b

    return pxf


def rotateImage(img, theta):
    h = img.shape[0]
    w = img.shape[1]
    center = (h/2, w/2)

    rotImg = np.zeros((h, w))

    cos = np.cos(np.radians(-theta))
    sin = np.sin(np.radians(-theta))
    rotDimMat = np.array(((cos, -sin), (sin, cos)))

    for i in range(w):
        for j in range(h):
            (x, y) = np.matmul(rotDimMat, np.array(
                [i - center[0], center[1] - j]))
            (x, y) = (x + center[0], center[1] - y)
            mask = (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
            if mask:
                rotImg[j][i] = GetBilinearPixel(img, x, y)
    return rotImg


if __name__ == "__main__":
    img = cv2.imread("Resources\The Leaning Tower of Pisa.jpg", 0)
    for _ in np.arange(3, 5, 0.1):
        _ = round(_, 1)
        cv2.imwrite(f"Output/Q05/Pisa_{_}_deg.jpg", rotateImage(img, _))
        print(f"Image rotated by {_} deg and saved in Output/Q05")
