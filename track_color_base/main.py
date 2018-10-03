# -*- coding:utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb


im_list = ['00000001.jpg', '1.jpeg', '2.jpg', '3.jpg', '4.jpg']

num_hist = 16
#bbox = (300, 142, 355, 288)
bbox = (246, 209, 270, 232)
bbox_w = bbox[2] - bbox[0]
bbox_h = bbox[3] - bbox[1]

surbox = (bbox[0] - int(bbox_w / 2.5),
          bbox[1] - int(bbox_h / 2.5),
          bbox[2] + int(bbox_w / 2.5),
          bbox[3] + int(bbox_h / 2.5))

im = cv2.imread(im_list[0])
#im = im[:, :, -1:]
sum_im = im.shape[0] * im.shape[1] *im.shape[2]
sum_bbox = bbox_w * bbox_h * 3

cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
cv2.rectangle(im, (surbox[0], surbox[1]), (surbox[2], surbox[3]), (0, 0, 255))

im_bbox = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
im_sur = im[surbox[1]:surbox[3], surbox[0]:surbox[2]]

im_hist = []
im_bbox_hist = []
im_sur_hist = []
#for i in range(3):
im_hist = cv2.calcHist([im], [0, 1, 2], None, [num_hist], [[0, 255], [0, 255], [0, 255]])

im_bbox_hist = cv2.calcHist([im_bbox], [0, 1, 2], None, [num_hist], [[0, 255], [0, 255], [0, 255]])

im_sur_hist = cv2.calcHist([im_sur], [0, 1, 2], None, [num_hist], [0, 255])



# cv2.imshow('hist', hist)
# draw hist
fig = plt.figure()
fig.add_subplot(331)
plt.title('im hist r')
plt.plot(range(len(im_hist[0])), im_hist[2])
fig.add_subplot(332)
plt.title('im hist g')
plt.plot(range(len(im_hist[0])), im_hist[1])
fig.add_subplot(333)
plt.title('im hist b')
plt.plot(range(len(im_hist[0])), im_hist[0])

fig.add_subplot(334)
plt.title('im_bbox hist r')
plt.plot(range(len(im_bbox_hist[0])), im_bbox_hist[2])
fig.add_subplot(335)
plt.title('im_bbox hist g ')
plt.plot(range(len(im_bbox_hist[0])), im_bbox_hist[1])
fig.add_subplot(336)
plt.title('im_bbox hist b')
plt.plot(range(len(im_bbox_hist[0])), im_bbox_hist[0])

fig.add_subplot(337)
plt.title('im_sur hist r')
plt.plot(range(len(im_sur_hist[0])), im_sur_hist[2])
fig.add_subplot(338)
plt.title('im_sur hist g ')
plt.plot(range(len(im_sur_hist[0])), im_sur_hist[1])
fig.add_subplot(339)
plt.title('im_sur hist b')
plt.plot(range(len(im_sur_hist[0])), im_sur_hist[0])


# show image
fig2 = plt.figure()
# plt.imshow(im, cmap='gray')

plt.imshow(im[:, :, ::-1])

# distractor aware
map_obj_sur = np.full(im.shape, 0.5 * 255).astype(np.uint8)

for h in range(im.shape[0]):
    for w in range(im.shape[1]):
        for c in range(im.shape[2]):
            #print im[h][w][c]
            index = int(im[h][w][c]/(256/num_hist))
            if 0 == im_bbox_hist[c][index]:
                element = 0
            else:
                element = (im_bbox_hist[c][index]) / (im_sur_hist[c][index])

            map_obj_sur[h][w][c] = np.uint8(element*255)


#print 'hist', im_bbox_hist

fig3 = plt.figure()
plt.title('obj surrounding map')
plt.imshow(np.sum(map_obj_sur[:, :, ::-1], axis=2), cmap='gray')


plt.show()

# cv2.imshow('win0', im)
# cv2.imshow('bbox', im_bbox)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


