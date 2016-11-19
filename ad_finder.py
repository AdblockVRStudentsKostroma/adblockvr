# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2
import time

imgs = [] # картинки с рекламой, которые нам надо вырезать
seeds = [] # массив ключевых точек и дескрипторов
detector = cv2.xfeatures2d.SIFT_create(1000) # инициализируем класс поиска ключевых точек
matcher = cv2.BFMatcher() # поиск соответствия по брутфорс перебору

# вспомогательный класс таймера, для подсчета сколько времени заняла нужная операция
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

##############################################################################################
# создает белый битмап заданного размера
def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color

    return image

##############################################################################################
# отсеивает найденные соответствия если они находятся друг в друге
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
#def remove_noise

##############################################################################################
# обработка найденной реклама на картинке делается здесь
def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    # TODO - блур рекламы на видеопотоке
    width, height = 960, 540
    x_offset = 0
    y_offset = 0

    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img2)

#TODO - функция снятия фрагмента изображения с экрана и пометки как рекламы
#def mark_ad

##############################################################################################
# находит наиболее подходящую рекламу по всем картинкам с рекламой
def find_match(kp, desc):
    max_matches = 0
    match_img = imgs[0]
    match_desc= seeds[0]

    for i, seed in enumerate(seeds):
        #поиск лучших соответствий
        raw_matches = matcher.knnMatch(desc, trainDescriptors = seed[1], k = 2)
        p1, p2, kp_pairs = filter_matches(kp, seed[0], raw_matches)
        if len(p1) > max_matches:
            max_matches = len(p1)
            match_desc = seed
            match_img = imgs[i]

    return (match_desc, match_img)
        

##############################################################################################
# находит рекламу и обрабатывает ее (по предварительно найденным точкам, для экономии времени)
# по сути мы предполагаем, что после предыдущего поиска ключевых точек, они все остались на
# экране, но только сместились
def match_and_draw(win, img1, img2, kp1, kp2, desc1, desc2):
    #поиск лучших соответствий
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
    #отсеиваем вложенные и пересекающиеся
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

    if len(p1) >= 4:# если четырехугольник
        # находим вычисленную трансформацию перспективы
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, None
    # мылим или еще как-то обрабатываем то что мы нашли
    explore_match(win, img1, img2, kp_pairs, status, H)

##############################################################################################
# основная функция, которая все делает (отображет окно с заблуренной и обработанной рекламой).
# точка входа в программу
##############################################################################################
def detect(Source):
    
    epsilon = 0.001
    long_detecting_cnt = 0
    total_time = 0

    fullpath = os.getcwd() + '\\images'
    names_img = os.listdir(fullpath)
    for name in names_img:
        img_path = os.path.join(fullpath, name)
        if os.path.isfile(img_path):
            imgs.append(cv2.imread(img_path, 0))
    #imgs.append(cv2.imread("images/cola.jpg", 0))
    
    for i in imgs:
        k, d = detector.detectAndCompute(i, None)
        seeds.append((k,d))
        #TODO: отображения проецнтов загрузки seeds     
    
    isCameraInput = Source is None
    
    if isCameraInput:
        cap = cv2.VideoCapture(0) # захват видеопотока с камеры
    else:
        cap = cv2.VideoCapture(Source) # захват видеопотока с файла

    count = 0
    ext = False
    
    cap.set(3,960)
    cap.set(4,540)
    
    while (not ext):
        ret, frame = cap.read()
        if (ret is not None) and (frame is not None):
            with Timer() as t:
                if count % 8 == 0: # т.к. detectAndCompute очень тяжелая, то делаем ее реже
                    kp2, desc2 = detector.detectAndCompute(frame, None)
                    (kp1, desc1), img1 = find_match(kp2, desc2)

            if desc2 is not None:
                match_and_draw('AdBlockVR', img1, frame, kp1, kp2, desc1, desc2)
                count += 1
                if t.interval>=epsilon:
                   print('Try to find match. Took %.03f sec.' % t.interval)
                   total_time += t.interval
                   long_detecting_cnt += 1
        elif isCameraInput is False:
            ext = True
        else:
            print("No frame retrieved, do you wish to continue?Y/N")
            pressed = input() # waitKey не хочет ждать поэтому юзаю стандартный ввод питона
            if ((pressed == 'N') or (pressed == 'n')):
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
