import cv2
import cvzone
import numpy as np
from tkinter import filedialog as fd

selectedSticker = 0
out = 0
fecharCamera = False
capturouCamera = False

img_counter = 0

cap = cv2.VideoCapture(0)

pathImg = 'user.png'

background = cv2.imread('background.png')
stickerGlasses = cv2.imread('eyeglasses.png', cv2.IMREAD_UNCHANGED)
stickerStar = cv2.imread('star.png', cv2.IMREAD_UNCHANGED)
stickerHeart = cv2.imread('heart.png', cv2.IMREAD_UNCHANGED)
stickerLightning = cv2.imread('lightning.png', cv2.IMREAD_UNCHANGED)
stickerSmile = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)
webCamIcon = cv2.imread('webcam.png', cv2.IMREAD_UNCHANGED)
fileIcon = cv2.imread('file-png.png', cv2.IMREAD_UNCHANGED)
userImage = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
capturar = cv2.imread('capturar.png', cv2.IMREAD_UNCHANGED)
fechar = cv2.imread('fechar.png', cv2.IMREAD_UNCHANGED)

resizedfilteredImageOriginal = cv2.resize(userImage, (75, 75), interpolation = cv2.INTER_AREA)

resizedUserImage = cv2.resize(userImage, (550, 550), interpolation = cv2.INTER_AREA)

def grayScaleFilter(image):
    filterResult = image
    for i in range(filterResult.shape[0]):
    	for j in range(filterResult.shape[1]):		
            mediaPond = filterResult.item(i,j,0) * 0.07 + filterResult.item(i,j,1) * 0.71 + filterResult.item(i,j,2) * 0.21
            filterResult.itemset((i,j,0),mediaPond) # canal B
            filterResult.itemset((i,j,1),mediaPond) # canal G
            filterResult.itemset((i,j,2),mediaPond) # canal R
    
    return filterResult

filteredImage = grayScaleFilter(userImage)

resizedfilteredImageGrayScale = cv2.resize(filteredImage, (75, 75), interpolation = cv2.INTER_AREA)

resizeWebCamIcon = cv2.resize(webCamIcon, (40, 40), interpolation = cv2.INTER_AREA)

resizeFilePng = cv2.resize(fileIcon, (40, 40), interpolation = cv2.INTER_AREA)


def overlay(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2


    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    out = background.copy()
    return out

def mouse_click(event, x, y, flags, param):
    global selectedSticker, out, pathImg
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= 227 and x <= 309 and y >= 34 and y <= 54:
            selectedSticker = stickerGlasses
        elif x >= 330 and x <= 408 and y >= 10 and y <= 79:
            selectedSticker = stickerStar
        elif x >= 451 and x <= 493 and y >= 5 and y <= 82:
            selectedSticker = stickerLightning
        elif x >= 529 and x <= 608 and y >= 9 and y <= 78:
            selectedSticker = stickerHeart
        elif x >= 629 and x <= 710 and y >= 5 and y <= 88:
            selectedSticker = stickerSmile
        elif x > 128 and x < 680 and y > 108 and y < 661:
            out = overlay(out, selectedSticker, x - 40, y - 40)            
        elif x > 83 and x < 158 and y > 690 and y < 765:
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)        
            out = overlay(out, resizedUserImageOriginal, 130, 110)
        elif x > 163 and x < 238 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            imgBlurred5x5 = cv2.blur(resizedUserImageOriginal,(10,10))      
            out = overlay(out, imgBlurred5x5, 130, 110)
        elif x > 243 and x < 318 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            imgGaussianBlurred5x5 = cv2.GaussianBlur(resizedUserImageOriginal,(5,5),0)     
            out = overlay(out, imgGaussianBlurred5x5, 130, 110)
        elif x > 323 and x < 398 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            erode = cv2.erode(resizedUserImageOriginal, kernel)   
            out = overlay(out, erode, 130, 110)
        elif x > 403 and x < 478 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilate = cv2.dilate(resizedUserImageOriginal, kernel)  
            out = overlay(out, dilate, 130, 110)
        elif x > 483 and x < 558 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            brightness = cv2.convertScaleAbs(resizedUserImageOriginal, beta=60)
            out = overlay(out, brightness, 130, 110)
        elif x > 563 and x < 638 and y > 690 and y < 765: 
            userImageOriginal = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
            resizedUserImageOriginal = cv2.resize(userImageOriginal, (550, 550), interpolation = cv2.INTER_AREA)  
            box = cv2.boxFilter(resizedUserImageOriginal, -1, (3,3))
            out = overlay(out, box, 130, 110)
        elif x > 643 and x < 718 and y > 690 and y < 765: 
            grayScaleFilter(resizedUserImage)
            out = overlay(out, resizedUserImage, 130, 110)
        elif x > 739 and x < 780 and y > 708 and y < 752:
            showCamera()
        elif x > 24 and x < 56 and y > 711 and y < 750:
            openFile()
        cv2.imshow('TrabalhoGB', out)
    
def mouse_click_camera(event, x, y, flags, param):   
    global fecharCamera, capturouCamera
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= 451 and x <= 518 and y >= 2 and y <= 27:
            capturouCamera = True
        elif x >= 52 and x <= 118 and y >= 2 and y <= 27:
            fecharCamera = True

out = overlay(background, stickerGlasses, 230, 5)
out = overlay(out, stickerStar, 330, 5)
out = overlay(out, stickerLightning, 430, 5)
out = overlay(out, stickerHeart, 530, 5)
out = overlay(out, stickerSmile, 630, 5)
out = overlay(out, resizedUserImage, 130, 110)
out = overlay(out, resizeWebCamIcon, 740, 710)
out = overlay(out, resizeFilePng, 20, 710)

out = overlay(out, resizedfilteredImageOriginal, 83, 690)

imgBlurred5x5 = cv2.blur(resizedfilteredImageOriginal,(5,5))
out = overlay(out, imgBlurred5x5, 163, 690)

imgGaussianBlurred5x5 = cv2.GaussianBlur(resizedfilteredImageOriginal,(5,5),0)
out = overlay(out, imgGaussianBlurred5x5, 243, 690)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
erode = cv2.erode(resizedfilteredImageOriginal, kernel)
out = overlay(out, erode, 323, 690)

dilate = cv2.dilate(resizedfilteredImageOriginal, kernel)
out = overlay(out, dilate, 403, 690)

brightness = cv2.convertScaleAbs(resizedfilteredImageOriginal, beta=60)
out = overlay(out, brightness, 483, 690)

box = cv2.boxFilter(resizedfilteredImageOriginal, -1, (3,3))
out = overlay(out, box, 563, 690)

out = overlay(out, resizedfilteredImageGrayScale, 643, 690)

out = cv2.putText(out, 'Stickers', (80, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

def openFile():
    file = fd.askopenfilename()
    if file:
        reRender(file)

def reRender(pathImage):
    global resizedUserImage, resizedfilteredImageOriginal, resizedfilteredImageGrayScale, out, pathImg
    pathImg = pathImage
    newImage = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)   
    resizedfilteredImageOriginal = cv2.resize(newImage, (75, 75), interpolation = cv2.INTER_AREA)  
    resizedUserImage = cv2.resize(newImage, (550, 550), interpolation = cv2.INTER_AREA)  
    
    filteredImage = grayScaleFilter(newImage)
    resizedfilteredImageGrayScale = cv2.resize(filteredImage, (75, 75), interpolation = cv2.INTER_AREA)  

    out = overlay(background, stickerGlasses, 230, 5)
    out = overlay(out, stickerStar, 330, 5)
    out = overlay(out, stickerLightning, 430, 5)
    out = overlay(out, stickerHeart, 530, 5)
    out = overlay(out, stickerSmile, 630, 5)
    out = overlay(out, resizedUserImage, 130, 110)
    out = overlay(out, resizeWebCamIcon, 740, 710)
    out = overlay(out, resizeFilePng, 20, 710)

    out = overlay(out, resizedfilteredImageOriginal, 83, 690)

    imgBlurred5x5 = cv2.blur(resizedfilteredImageOriginal,(5,5))
    out = overlay(out, imgBlurred5x5, 163, 690)

    imgGaussianBlurred5x5 = cv2.GaussianBlur(resizedfilteredImageOriginal,(5,5),0)
    out = overlay(out, imgGaussianBlurred5x5, 243, 690)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode = cv2.erode(resizedfilteredImageOriginal, kernel)
    out = overlay(out, erode, 323, 690)

    dilate = cv2.dilate(resizedfilteredImageOriginal, kernel)
    out = overlay(out, dilate, 403, 690)

    brightness = cv2.convertScaleAbs(resizedfilteredImageOriginal, beta=60)
    out = overlay(out, brightness, 483, 690)

    box = cv2.boxFilter(resizedfilteredImageOriginal, -1, (3,3))
    out = overlay(out, box, 563, 690)

    out = overlay(out, resizedfilteredImageGrayScale, 643, 690)

    out = cv2.putText(out, 'Stickers', (80, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('TrabalhoGB', out)


def showCamera():
    global cap, fecharCamera, capturouCamera, pathImg, out, userImage
    fecharCamera = False
    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (550, 550), interpolation = cv2.INTER_AREA)
        frame = cvzone.overlayPNG(frame, fechar, [50, 1])
        frame = cvzone.overlayPNG(frame, capturar, [450, 1])
        cv2.imshow('Camera', frame)
        cv2.setMouseCallback('Camera', mouse_click_camera)
        
        cv2.waitKey(1)

        if capturouCamera:
            cv2.imwrite('cameraCapture.png', frame)
            reRender('cameraCapture.png')
            capturouCamera = False
        if fecharCamera:
            break

    cv2.destroyWindow('Camera')

cv2.imshow('TrabalhoGB', out)
cv2.setMouseCallback('TrabalhoGB', mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()