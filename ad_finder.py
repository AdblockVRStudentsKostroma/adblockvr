# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2
import time
import image_cutter as im_cut

imgs = [] # картинки с рекламой, которые нам надо вырезать
seeds = [] # массив ключевых точек и дескрипторов
detector = cv2.xfeatures2d.SIFT_create(1000) # инициализируем класс поиска ключевых точек
matcher = cv2.BFMatcher() # поиск соответствия по брутфорс перебору
width, height = 800, 600
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

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


# Функция определения определен шум или нет, roi_corners - точки, w - ширина, h - высота
# coeff0 - процент минимального расстояния
# coeff1 - процент максимального расстояния
# coeff2 - процент максимального отношения сторон
def is_noise(roi_corners, w,h, coeff0, coeff1,coeff2):

    diag = w*w+h*h
    minSize=diag*coeff0
    maxSize=diag*coeff1
    minLen=1000000000
    maxLen=0
    delta=0
    for i in range(0,2):
        for j in range(i+1,3):
            xDiff = min(abs(roi_corners[i][0]-roi_corners[j][0]),diag+100)
            yDiff = min(abs(roi_corners[i][1]-roi_corners[j][1]),diag+100)
            len = xDiff*xDiff+yDiff*yDiff
            minLen = min(minLen, len)
            maxLen = max(minLen, len)
    delta = maxLen/(minLen+0.00001)
    result = (minLen>minSize) and (maxLen<maxSize) and (delta<coeff2)
    return not result


##############################################################################################
# обработка найденной реклама на картинке делается здесь
def explore_match(win, img2, ads):
    mask = np.zeros(img2.shape, dtype=np.uint8)
    h2, w2 = img2.shape[:2]
    for ad in ads:
        if (ad[3] is not None)and(ad[0] is not None) and (img2 is not None):
            h1, w1 = ad[0].shape[:2]
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), ad[3]).reshape(-1, 2))
        
            roi_corners = np.array([[(corners[0][0],corners[0][1]), 
                (corners[1][0],corners[1][1]), 
                (corners[2][0],corners[2][1]), 
                (corners[3][0], corners[3][1])]], dtype=np.int32)
            if not is_noise(roi_corners[0],h1,w1, 0.0001, 0.9, 3.0) and cv2.isContourConvex(roi_corners):
                white = (255, 255, 255)
                cv2.fillConvexPoly(mask, roi_corners, white)
    
    # apply the mask
    masked_image = cv2.bitwise_and(img2, mask)

    #########################
    ## blurred_image = cv2.boxFilter(img2, -1, (27, 27)) - убирает контуры
    #########################
    blurred_image = cv2.boxFilter(img2, -1, (27, 27))
    img2 = img2 + (cv2.bitwise_and((blurred_image-img2), mask))

    # view params
    x_offset = 0
    y_offset = 0
#TODO: Отрисовать UI
    height, width = img2.shape[:2]

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE) # запрет изменять размер окна
    cv2.resizeWindow(win, width, height) # задание размера окна
    font = cv2.FONT_HERSHEY_SIMPLEX # шрифт текста
    cv2.putText(img2, 'Press Q to exit', (10,20), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img2, 'Press C to point advertisement', (10,50), font, 0.75, (255,0,0), 2, cv2.LINE_AA)# кнопки для выхода
    cv2.imshow(win, img2)
    #out.write(img2)

#TODO - функция снятия фрагмента изображения с экрана и пометки как рекламы
#def mark_ad

##############################################################################################
# находит подходящую рекламу по всем картинкам с рекламой
def find_match(kp, desc):
    result = []    
    for i, seed in enumerate(seeds):
        #поиск лучших соответствий
        raw_matches = matcher.knnMatch(desc, trainDescriptors = seed[1], k = 2)
        p1, p2, kp_pairs = filter_matches(kp, seed[0], raw_matches)
        if len(p1) > len(seed[0])*0.025: #Процент точности
            result.append((seed, imgs[i]))
    return result
        

##############################################################################################
# находит рекламу и обрабатывает ее (по предварительно найденным точкам, для экономии времени)
# по сути мы предполагаем, что после предыдущего поиска ключевых точек, они все остались на
# экране, но только сместились
def match_and_draw(win, kdi, img2, kp2, desc2):
    #поиск лучших соответствий
    ads = []
    for o in kdi:
        raw_matches = matcher.knnMatch(o[0][1], trainDescriptors = desc2, k = 2)
        #отсеиваем вложенные и пересекающиеся
        p1, p2, kp_pairs = filter_matches(o[0][0], kp2, raw_matches)
    
        if len(p1) >= 4:# если четырехугольник
            # находим вычисленную трансформацию перспективы
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        else:
            H, status = None, None
        ads.append((o[1], kp_pairs, status, H))
    # мылим или еще как-то обрабатываем то что мы нашли
    explore_match(win, img2, ads)

##############################################################################################
# основная функция, которая все делает (отображет окно с заблуренной и обработанной рекламой).
# точка входа в программу
##############################################################################################
def detect(Source):
    
    epsilon = 0.001
    long_detecting_cnt = 0
    total_time = 0

    fullpath = os.getcwd() + '\\images' # полный путь к директории с рекламой
    names_img = os.listdir(fullpath) # список файлов и папок в каталоге с рекламой
    for name in names_img:
        img_path = os.path.join(fullpath, name) # полный путь до файлов и папок в каталоге с рекламой
        if os.path.isfile(img_path): # определение, является ли файлом
            imgs.append(cv2.imread(img_path)) # добавление изображений с рекламой в массив

    print('''Loading advertisement's images''')
    for i in imgs:
        k, d = detector.detectAndCompute(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY), None)
        seeds.append((k,d))
        #TODO: отображения проецнтов загрузки seeds     
    
    isCameraInput = Source is None
    
    if isCameraInput:
        cap = cv2.VideoCapture(0) # захват видеопотока с камеры
    else:
        cap = cv2.VideoCapture(Source) # захват видеопотока с файла

    count = 0
    ext = False
    
    cap.set(3,width)
    cap.set(4,height)
    
    while (not ext):
        ret, frame = cap.read()
        if (ret is not None) and (frame is not None):
            with Timer() as t:
                if count % 8 == 0: # т.к. detectAndCompute очень тяжелая, то делаем ее реже
                    kp2, desc2 = detector.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), None)
                    kdi = find_match(kp2, desc2)

            if desc2 is not None:
                match_and_draw('AdBlockVR', kdi, frame, kp2, desc2)
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

        pressed = cv2.waitKey(10) & 0xFF
        if ((pressed == ord('q')) or (pressed == ord('Q'))):
            ext = True
        elif ((pressed == ord('c')) or (pressed == ord('C'))):
            new_img_fname = im_cut.cut_image(frame)
            if new_img_fname is not None:
                imgs.append(cv2.imread(new_img_fname))
                k, d = detector.detectAndCompute(cv2.cvtColor(imgs[-1],cv2.COLOR_BGR2GRAY), None)
                seeds.append((k,d))

            #TODO - обработка нажатий клавиатуры
            
    if long_detecting_cnt != 0:
        print("Total tries  %.i" % long_detecting_cnt)
        print("Average time %.03f" % (total_time/long_detecting_cnt))
    
    cap.release()
    cv2.destroyAllWindows()
    #out.release()
