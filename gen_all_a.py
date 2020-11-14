import os
import math

from skimage import io, transform, img_as_float
import numpy as np
import cv2


numImgs = 1
# numGpus = 16
numGpus = 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
img_1 = io.imread('data/selfie.jpg')
img_2 = io.imread('data/0_target.jpg')
img_1 = img_1/255
img_2 = img_2/255

img_1_mask = io.imread('data/selfie_mask.jpg')      #  mask_rcnn生成  单通道
img_1_mask =img_1_mask/255


re_h = 400     # 自拍图缩放后大小
re_w = 270
t_h = 300      # 自拍图在背景图上位置
t_w = 10


def onetothree(one):
    rows_t, cols_t= one.shape
    a = np.zeros((rows_t, cols_t, 3))
    for i in range(3):
        a[:, :, i] = one
    return a


def Padding_mask(mask_img, target_img, re_h, re_w, t_h, t_w):
    self_img_c = transform.resize(mask_img, [re_h, re_w])
    rows_t, cols_t, h_t = target_img.shape
    p = np.zeros((rows_t, cols_t))
    p[t_h:t_h + re_h, t_w:t_w + re_w] = self_img_c
    for i in range(rows_t):
        for j in range(cols_t):
            if (p[i,j]>0):
                p[i,j]=1
            else:
                p[i,j]=0
    io.imsave('data/0_c_mask.jpg', p)
    return p



def Padding(self_img, target_img, re_h, re_w, t_h, t_w):
    self_img_c = transform.resize(self_img, [re_h, re_w])
    rows_t, cols_t, h_t = target_img.shape
    g = np.zeros((rows_t, cols_t, h_t))
    g[t_h:t_h + re_h, t_w:t_w + re_w, :] = self_img_c
    return g



def mask_dilate(mask_img):
    img1_1 = img_as_float(mask_img)
    h = img1_1.shape[0]
    w = img1_1.shape[1]
    h1 = h
    w1 = w
    if h!=700 and w!=700:
        if h > w:
            h1 = 700
            w1 = int((h1 * w) / h)
        else:
            w1 = 700
            h1 = int((w1 * h) / w)
    img1_1 = transform.resize(img1_1, (h1, w1))
    r = 35
    img1_2 = cv2.GaussianBlur(img1_1, (r,r) ,r/3)
    img1_3 = img1_2
    for i in range(h1):
        for j in range(w1):
            if (img1_2[i,j]>0.1):
                img1_3[i,j]=1
            else:
                img1_3[i,j]=0
    io.imsave('data/0_c_mask_dilated.jpg' , img1_3)
    return img1_3


def tmix(target_img, p_img, mask_p_img):
    m = target_img * (1-mask_p_img) + p_img * mask_p_img
    io.imsave('data/0_naive.jpg', m)
    return m

# 生成c_mask
p_m = Padding_mask(img_1_mask, img_2, re_h, re_w, t_h, t_w)
# 生成c_mask_dilated
d = mask_dilate(p_m)
# 生成naive
p_s = Padding(img_1, img_2, 400, 270, 300, 10)
p_m_t = onetothree(p_m)
m = tmix(img_2, p_s, p_m_t)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if os.path.exists('results') == 0:
	os.mkdir('results')

N = int(math.ceil(float(numImgs)/numGpus))

for j in range(1, numGpus+1):
	cmd = ''
	for i in range(1, N+1):
		idx = (i-1) * numGpus + (j-1)
		if idx >= 0 and idx < numImgs:
			print('Working on image idx = ', idx)
			part_cmd1 =' /content/torch/install/bin/th neural_gram.lua '\
					   ' -content_image data/' + str(idx) + '_naive.jpg  '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
					   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
					   ' -gpu ' + str(j-1) + ' -original_colors 0 -image_size 700 '\
					   ' -output_image  results/' + str(idx) + '_inter_res.jpg'\
					   ' -print_iter 100 -save_iter 100 && '
			part_cmd2 =' /content/torch/install/bin/th neural_paint.lua '\
					   ' -content_image data/' + str(idx) + '_naive.jpg '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
					   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
					   ' -cnnmrf_image  results/' + str(idx) + '_inter_res.jpg  '\
					   ' -gpu ' + str(j-1) + ' -original_colors 0 -image_size 700 '\
					   ' -index ' + str(idx) + ' -wikiart_fn data/wikiart_output.txt '\
					   ' -output_image  results/' + str(idx) + '_final_res.jpg' \
					   ' -print_iter 100 -save_iter 100 '\
					   ' -num_iterations 1000 &&'
			cmd = cmd + part_cmd1 + part_cmd2
	cmd = cmd[1:len(cmd)-1]
	print(cmd)
	os.system(cmd)
