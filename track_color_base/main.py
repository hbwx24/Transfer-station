# -*- coding:utf-8 -*-

import cv2
import matplotlib.pyplot as plt

im_list = ['1.jpeg', '2.jpg', '3.jpg', '4.jpg']
bbox = (290, 132, 365, 298)

im = cv2.imread(im_list[0], 0)
# cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255))
hist = cv2.calcHist([im], [0], None, [256], [0, 255])
# cv2.imshow('hist', hist)

im_bbox = im[132:298, 290:365]
hist_bbox = cv2.calcHist([im_bbox], [0], None, [256], [0, 255])



cv2.imshow('win0', im)
cv2.imshow('bbox', im_bbox)

figure = plt.figure()
f1 = figure.add_subplot(221)
plt.title('im hist')
f1.plot(range(len(hist)), hist)

f2 = figure.add_subplot(222)
plt.title('bbox hist')
f2.plot(range(len(hist_bbox)), hist_bbox)


print im
plt.show(figure)
# print hist
# cv2.waitKey(0)
cv2.destroyAllWindows()
