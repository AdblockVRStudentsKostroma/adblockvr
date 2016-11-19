# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

imgs = []
seeds = []
detector = cv2.xfeatures2d.SIFT_create(1000)
matcher = cv2.BFMatcher()

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color

    return image

def filter_matches(kp1, kp2, matches, ratio = 0.65):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


# TODO - Bool функция определения определен шум или нет


def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    # TODO - блур рекламы на видеопотоке
    width, height = 960, 540
    x_offset = 0
    y_offset = 0

    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img2)

 #TODO - функция снятия фрагмента изображения с экрана и пометки как рекламы


def detect(Source):
    epsilon = 0.001
    long_detecting_cnt = 0
    total_time = 0

    #TODO  - чтение картинок - образов рекламы в img[] - автоматизированно
    imgs.append(cv2.imread("images/cola.jpg", 0))
    
    for i in imgs:
        k, d = detector.detectAndCompute(i, None)
        seeds.append((k,d))
        #TODO: отображения проецнтов загрузки seeds     

    def find_match(kp, desc):
        max_matches = 0
        match_img = imgs[0]
        match_desc= seeds[0]

        for i, seed in enumerate(seeds):
            raw_matches = matcher.knnMatch(desc, trainDescriptors = seed[1], k = 2)
            p1, p2, kp_pairs = filter_matches(kp, seed[0], raw_matches)
            if len(p1) > max_matches:
                max_matches = len(p1)
                match_desc = seed
                match_img = imgs[i]

        return (match_desc, match_img)
            

    def match_and_draw(win, img1, img2, kp1, kp2, desc1, desc2):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        else:
            H, status = None, None

        explore_match(win, img1, img2, kp_pairs, status, H)
    
    if Source is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(Source)

    count = 0
    ext = False
    
    cap.set(3,960)
    cap.set(4,540)
    
    while (not ext):
        ret, frame = cap.read()
        if (ret is not None) and (frame is not None):
            with Timer() as t:
                if count % 8 == 0:
                    kp2, desc2 = detector.detectAndCompute(frame, None)
                    (kp1, desc1), img1 = find_match(kp2, desc2)

            if desc2 is not None:
                match_and_draw('AdBlockVR', img1, frame, kp1, kp2, desc1, desc2)
                count += 1
                if t.interval>=epsilon:
                   print('Try to find match. Took %.03f sec.' % t.interval)
                   total_time += t.interval
                   long_detecting_cnt += 1
        else:
            print("Couldn't get input capture.")
            ext = True

        pressed = cv2.waitKey(70) & 0xFF
        if ((pressed == ord('q')) or (pressed == ord('Q'))):
            ext = True

            #TODO - обработка нажатий клавиатуры
            
    if long_detecting_cnt != 0:
        print("Total tries  %.i" % long_detecting_cnt)
        print("Average time %.03f" % (total_time/long_detecting_cnt))
    cap.release()
    cv2.destroyAllWindows()
