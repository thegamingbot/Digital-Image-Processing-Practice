import cv2


if __name__ == "__main__":
    img = cv2.imread('Resources/pout-dark.jpg', 0)
    equ = cv2.equalizeHist(img)
    cv2.imwrite('Output/Q06/equalize.png', equ)
    print("Output image generated at Output/Q06.")
