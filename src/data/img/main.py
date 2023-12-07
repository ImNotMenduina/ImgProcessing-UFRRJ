import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# install bin on Windows
PATH = r"C:\Users\mynam\AppData\Local\Programs\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = PATH + r"\tesseract.exe"

from googletrans import Translator
from matplotlib import pyplot as plt


def translator(img):
    text = pytesseract.image_to_string(img, lang="fra")
    lines = text.splitlines()

    # print(text)
    # print(lines)

    finalText = ""

    for b in lines:
        if (b != ''):
            finalText += b + ' '

    print(finalText)
    translator = Translator()
    translation = translator.translate(finalText, src="fr", dest="pt")

    return translation.text


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


def simpleThresholding(img):

    return cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)


def filtroMediana(img, k):
    # imgArray = [imgOriginal, imgReplicate] #HERE I STORED BOTH IMAGES
    # title = ["Original", "Filtro da Mediana"]
    #
    # showMultipleImages(imgArray, title, (12,8),2, 1)

    return cv2.medianBlur(img, k)


def dilate(img):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    return img


def loadImg(src):
    PATH = "{}{}".format("..\\data\\img\\", src)
    imgOriginal = cv2.imread(PATH)  # ORIGINAL IMG

    return imgOriginal


def wordBoxes(img):
    # imH, imW,_ = img.shape
    boxes = pytesseract.image_to_data(img)

    for x, linha in enumerate(boxes.splitlines()):
        if x != 0:
            linha = linha.split()
            if len(linha) == 12:
                ### linha with length equal to 12, it means that there's a word.
                x, y, w, h = int(linha[6]), int(linha[7]), int(linha[8]), int(linha[9])
                # palavra = linha[11]
                cv2.rectangle(img, (x, y), (w + x, h + y), (255, 0, 0), 2)
                # cv2.putText(img, palavra, (x, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    return img


def main():
    # load image : text in french
    imgOriginal = loadImg("textEnglish.png")

    # 2 grayscale
    img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # median filter: treat noise
    img = filtroMediana(img, 3)

    # imagem com filtro black hat
    imgBlackhatFilter = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((15, 15), np.uint8))

    #cv2.imshow("gomem", imgBlackhatFilter)
    #ret, imgThresholded = simpleThresholding(imgBlackhatFilter)

    # dilate: we want to treat some characters and recover them
    img = dilate(imgBlackhatFilter)

    ret, imgThresholded = simpleThresholding(imgBlackhatFilter)

    # median filter: treat noise
    img = filtroMediana(img, 3)

    # translate our text
    text = translator(img)

    ###recognizing words
    imgOriginal = wordBoxes(imgOriginal)
    img = wordBoxes(imgThresholded)

    ##compare before and after
    images = [imgOriginal, img]
    titles = ["Before", "After"]
    showMultipleImages(images, titles, (20, 8), 2, 1)

    cv2.imwrite("englishOriginal.png", imgOriginal)
    cv2.imwrite("english.png" ,img)

    # translate
    print(text)


main()