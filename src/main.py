import cv2
import numpy as np

def showImage(img):
    from matplotlib import pyplot as plt
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv uses BGR, but matplot uses RGB
    plt.imshow(img)  # prepare the matrix
    plt.show()  # show the matrix

def main():
    img = cv2.imread(r"..\data\img\train.jpg")
    showImage(img)

main()