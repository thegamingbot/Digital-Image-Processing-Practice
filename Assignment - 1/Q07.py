import cv2
from skimage import exposure


if __name__ == "__main__":
    reference = cv2.imread('Resources/pout-dark.jpg', 0)
    image = cv2.imread('Resources/pout-bright.jpg', 0)

    matched = exposure.match_histograms(image, reference, multichannel=True)

    cv2.imwrite('Output/Q08/matched.png', matched)
    print(f"Output image generated at Output/Q07.")
