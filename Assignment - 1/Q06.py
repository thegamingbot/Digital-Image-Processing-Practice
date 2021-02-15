import cv2


if __name__ == "__name__":
    img = cv2.imread('Resources/pout-dark.jpg', 0)
    equ = cv2.equalizeHist(img)
    cv2.imwrite('Output/Q07/equalize.png', equ)
    print(f"Output image generated at Output/Q06.")
