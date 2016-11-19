# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import uuid

img_pt = []
name_sv_img = 'images\\' + str(uuid.uuid4()) + '.jpg'
img_ = None
clone_img = None
isNotSave = True

def select_image(event,x,y,flags,param):
    global name_sv_img
    name_sv_img = 'images\\' + str(uuid.uuid4()) + '.jpg'
    if event == cv2.EVENT_LBUTTONDOWN:
        img_pt.append(x)
        img_pt.append(y)

    if event == cv2.EVENT_LBUTTONUP:
        img_pt.append(x)
        img_pt.append(y)
        global img_
        crop_img = img_[min(img_pt[1],img_pt[3]):max(img_pt[1],img_pt[3]),
                        min(img_pt[0],img_pt[2]):max(img_pt[0], img_pt[2])]
        cv2.rectangle(img_, (min(img_pt[0],img_pt[2])-2,min(img_pt[1],img_pt[3])-2),(max(img_pt[0], img_pt[2])+2,max(img_pt[1],img_pt[3])+2),(255,0,0), thickness=1, lineType=8, shift=0)
        cv2.imshow('Test', img_)
        pressed = cv2.waitKey(0) & 0xFF
        if ((pressed == ord('s')) or (pressed == ord('S'))):
            cv2.imwrite(name_sv_img, crop_img)
            global isNotSave
            isNotSave = False
        else:
            img_ = clone_img.copy()
            cv2.imshow('Test', img_)
            
        img_pt.clear()

def cut_image(frame):
    cv2.namedWindow('Test', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Test', 800, 600)
    global img_
    img_ = frame.copy()
    global clone_img
    clone_img = img_.copy()
    cv2.setMouseCallback('Test', select_image)

    while (isNotSave):
        cv2.imshow('Test', img_)
        pressed = cv2.waitKey(40) & 0xFF
        if ((pressed == ord('q')) or (pressed == ord('Q'))):
            global name_sv_img
            name_sv_img = None
            break


    cv2.destroyWindow('Test')
    global isNotSave
    isNotSave = True
    

    return name_sv_img
