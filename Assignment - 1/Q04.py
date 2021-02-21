import cv2 as cv
import numpy as np
import sys


def resizeInBuilt(img, scale):
    imgSize = tuple([int(x * scale) for x in img.shape[:2]])
    resizedImg = cv.resize(img, imgSize, interpolation=cv.INTER_LINEAR)
    cv.imwrite(f"Output/Q04/Lena_{scale}x_builtin.png", resizedImg)
    print(f"{scale}x built-in resized image stored at Output/Q04")


def GetBilinearPixel(imArr, posX, posY):
    out = []

    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi+1, imArr.shape[1]-1)
    modYiPlusOneLim = min(modYi+1, imArr.shape[0]-1)

    for chan in range(imArr.shape[2]):
        bl = imArr[modYi, modXi, chan]
        br = imArr[modYi, modXiPlusOneLim, chan]
        tl = imArr[modYiPlusOneLim, modXi, chan]
        tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]

        b = modXf * br + (1. - modXf) * bl

        t = modXf * tr + (1. - modXf) * tl
        pxf = modYf * t + (1. - modYf) * b
        out.append(int(pxf))

    return out


def scaleImg(im, scale):
    enlargedShape = list(
        map(int, [im.shape[0]*scale, im.shape[1]*scale, im.shape[2]]))
    enlargedImg = np.empty(enlargedShape, dtype=np.uint8)
    rowScale = float(im.shape[0]) / float(enlargedImg.shape[0])
    colScale = float(im.shape[1]) / float(enlargedImg.shape[1])

    for r in range(enlargedImg.shape[0]):
        for c in range(enlargedImg.shape[1]):
            orir = r * rowScale
            oric = c * colScale
            enlargedImg[r, c] = GetBilinearPixel(im, oric, orir)
    cv.imwrite(f"Output/Q04/Lena_{scale}x_custom.png", enlargedImg)
    print(f"{scale}x custom resized image stored at Output/Q04")


if __name__ == "__main__":
    im = cv.imread("Resources/Lena.png")
    if im is None:
        sys.exit("Could not read the image.")

    scaleImg(im, 0.5)
    scaleImg(im, 1)
    scaleImg(im, 2)

    resizeInBuilt(im, 0.5)
    resizeInBuilt(im, 1)
    resizeInBuilt(im, 2)
