import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import torch
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from model import *

TEST_IMAGE_PATH = './test_image.png'

'''
imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image
'''

def find_peaks(heatmap_avg, threshold=0.1):
    all_peaks = []
    num_peaks = 0
        
    for part in range():
        map_orig = heatmap_avg[:, :, part]
        map_filt = gaussian_filter(map_orig, sigma=3)
        
        map_L = np.zeros_like(map_filt)
        map_T = np.zeros_like(map_filt)
        map_R = np.zeros_like(map_filt)
        map_B = np.zeros_like(map_filt)
        map_L[1:, :] = map_filt[:-1, :]
        map_T[:, 1:] = map_filt[:, :-1]
        map_R[:-1, :] = map_filt[1:, :]
        map_B[:, :-1] = map_filt[:, 1:]
        
        peaks_binary = np.logical_and.reduce(
            (map_filt >= map_L, map_filt >= map_T,
             map_filt >= map_R, map_filt >= map_B,
             map_filt > thresh_1)
        )
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_ids = range(num_peaks, num_peaks + len(peaks))
        peaks_with_scores = [peak + (map_orig[peak[1], peak[0]],) for peak in peaks]
        peaks_with_scores_and_ids = [peaks_with_scores[i] + (peaks_ids[i],) \
                                     for i in range(len(peaks_ids))]
        all_peaks.append(peaks_with_scores_and_ids)
        num_peaks += len(peaks)

    return all_peaks, num_peaks

#image = image_loader(TEST_IMAGE_PATH)

pretrained_weights = torch.load('./hand/pose_iter_102000.caffemodel.pt')
model = HandPoseModel()
model.load_state_dict(pretrained_weights,strict=False)

def _pad_image(image, stride=1, padvalue=0):
    assert len(image.shape) == 2 or len(image.shape) == 3
    h, w = image.shape[:2]
    pads = [None] * 4
    pads[0] = 0 # left
    pads[1] = 0 # top
    pads[2] = 0 if (w % stride == 0) else stride - (w % stride) # right
    pads[3] = 0 if (h % stride == 0) else stride - (h % stride) # bottom
    num_channels = 1 if len(image.shape) == 2 else image.shape[2]
    image_padded = np.ones((h+pads[3], w+pads[2], num_channels), dtype=np.uint8) * padvalue
    image_padded = np.squeeze(image_padded)
    image_padded[:h, :w] = image
    
    return image_padded, pads

def process(model, image_orig,mode='heatmap'):
    scale = 368/image_orig.shape[1]
    scale = scale*2
    image =  cv2.resize(image_orig, (0,0), fx=scale, fy=scale)
    image_padded, pads = _pad_image(image, 8, 128)
    image_tensor = np.expand_dims(np.transpose(image_padded, (2, 0, 1)), 0)
    image_tensor = np.float32(image_tensor) / 255.0 - 0.5
    image_tensor = np.ascontiguousarray(image_tensor)
    image_tensor = torch.from_numpy(image_tensor).float()
    print(image_tensor.shape)

    with torch.no_grad():
        output = model(image_tensor).numpy()
    print(output)
    output = np.transpose(np.squeeze(output), (1, 2, 0))
    #heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    #output = output[:image_padded.shape[0] - pads[3], :image_padded.shape[1] - pads[2], :]
    output = cv2.resize(output, (image_orig.shape[1], image_orig.shape[0]))
    #heatmap_avg += (heatmap / len(multipliers))
    
    image_out = image_orig

    mask = np.zeros_like(image_orig).astype(np.float32)
    if mode == "heatmap":
        for chn in range(0, output.shape[-1]-2):
            m = np.repeat(output[:,:,chn:chn+1],3, axis=2)
            m = 255*( np.abs(m)>0.2)            
            mask = mask + m*(mask==0)
        mask = np.clip(mask, 0, 255)
        image_out = image_out*0.8 + mask*0.2
    else:
        peaksR = find_peaks(out, peaks_th, sigma=sigma)[0]
        peaksL = find_peaks(-out, peaks_th, sigma=sigma)[0]

        print(peaksR)
        for peak in peaksR:
            if(len(peak)):
                peak = peak[0]
                cv2.drawMarker(image_out, (peak[0], peak[1]), (0,255,0), cv2.MARKER_STAR )
    
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)
    
    return image_out

if __name__ == '__main__':
    print(model)
    image = cv2.imread(TEST_IMAGE_PATH)
    image = cv2.resize(image, (960,540))
    image_out = process(model, image)
    plt.figure(figsize=(12,12))
    plt.imshow(image_out)
    plt.show()

