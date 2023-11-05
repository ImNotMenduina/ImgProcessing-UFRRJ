import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# install bin on Windows
PATH = r"C:\Users\mynam\AppData\Local\Programs\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = PATH + r"\tesseract.exe"

from googletrans import Translator
from matplotlib import pyplot as plt

def tradutor():
    img = cv2.imread(r"..\data\img\textEnglish.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img, lang="eng")
    lines = text.splitlines()

    finalText = ""

    for b in lines:
        if (b != ''):
            finalText += b + ' '

    translator = Translator()
    translation = translator.translate(finalText, src="en", dest="pt")

    print(translation.text)

def showSingleImage(img, title, size):
    fig, axis = plt.subplots(figsize=size)

    axis.imshow(img, 'gray')
    axis.set_title(title, fontdict={'fontsize': 22, 'fontweight': 'medium'})
    plt.show()
def showMultipleImages(imgsArray, titlesArray, size, x, y):
    if (x < 1 or y < 1):
        print("ERRO: X e Y não podem ser zero ou abaixo de zero!")
        return
    elif (x == 1 and y == 1):
        showSingleImage(imgsArray, titlesArray)
    elif (x == 1):
        fig, axis = plt.subplots(y, figsize=size)
        yId = 0
        for img in imgsArray:
            axis[yId].imshow(img, 'gray')
            axis[yId].set_anchor('NW')
            axis[yId].set_title(titlesArray[yId], fontdict={'fontsize': 18, 'fontweight': 'medium'}, pad=10)

            yId += 1
    elif (y == 1):
        fig, axis = plt.subplots(1, x, figsize=size)
        fig.suptitle(titlesArray)
        xId = 0
        for img in imgsArray:
            axis[xId].imshow(img, 'gray')
            axis[xId].set_anchor('NW')
            axis[xId].set_title(titlesArray[xId], fontdict={'fontsize': 18, 'fontweight': 'medium'}, pad=10)

            xId += 1
    else:
        fig, axis = plt.subplots(y, x, figsize=size)
        xId, yId, titleId = 0, 0, 0
        for img in imgsArray:
            axis[yId, xId].set_title(titlesArray[titleId], fontdict={'fontsize': 18, 'fontweight': 'medium'}, pad=10)
            axis[yId, xId].set_anchor('NW')
            axis[yId, xId].imshow(img, 'gray')
            if (len(titlesArray[titleId]) == 0):
                axis[yId, xId].axis('off')

            titleId += 1
            xId += 1
            if xId == x:
                xId = 0
                yId += 1
    plt.show()

def filtroMediana():
    imgOriginal = cv2.imread(r"..\data\img\dog.jpg") #ORIGINAL IMG
    imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
    imgReplicate = cv2.medianBlur(imgOriginal, 7)
    imgArray = [imgOriginal, imgReplicate] #HERE I STORED BOTH IMAGES
    title = ["Original", "Filtro da Mediana"]
    showMultipleImages(imgArray, title, (12,8),2, 1)

def main():
    filtroMediana()
    #tradutor()

main()