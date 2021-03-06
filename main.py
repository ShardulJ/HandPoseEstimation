import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from IPython.display import clear_output
import torch
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from model import *

TEST_IMAGE_PATH = './test.png.jpg'

def find_peaks(heatmap_avg, threshold=0.1):
    all_peaks = []
    num_peaks = 0
        
    for part in range(22):
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
             map_filt > threshold)
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
    
    heatmap_avg = np.zeros((image_orig.shape[0], image_orig.shape[1], 22))
    scale = 368/image_orig.shape[0]
    scale = scale*2
    image =  cv2.resize(image_orig, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image_padded, pads = _pad_image(image, 8, 128)
    image_tensor = np.transpose(np.float32(image_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    image_tensor = np.ascontiguousarray(image_tensor)
    image_tensor = torch.from_numpy(image_tensor).float()

    with torch.no_grad():
        output = model(image_tensor)
    output = output.numpy()    

    output = np.transpose(np.squeeze(output), (1, 2, 0))
    output = cv2.resize(output, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    output = output[:image_padded.shape[0] - pads[2], :image_padded.shape[1] - pads[3], :]
    output = cv2.resize(output, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    heatmap_avg += output
    image_out = image_orig
    print(heatmap_avg)
    #something wrong below
    mask = np.zeros_like(image_out).astype(np.float32)
    if mode == "heatmap":
        for chn in range(0, heatmap_avg.shape[-1]):
            m = np.repeat(heatmap_avg[:,:,chn:chn+1],3, axis=2)
            m = 255*( np.abs(m)>0.2)            
            mask = mask + m*(mask==0)
        mask = np.clip(mask, 0, 255)
        image_out = image_out*0.8 + mask*0.2
    else:
        peaksR = find_peaks(heatmap_avg, 0.1)[0]
        peaksL = find_peaks(-heatmap_avg, 0.1)[0]

        print(peaksR)
        for peak in peaksR:
            if(len(peak)):
                peak = peak[0]
                cv2.drawMarker(image_out, (peak[0], peak[1]), (0,255,0), cv2.MARKER_STAR )
    
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)
    
    return image_out

def load_weights(model, state_dict):
    model_state_dict = {}
    for name in model.state_dict().keys():
        if len(name.split('.')) == 3:
            model_state_dict[name] = state_dict['.'.join(name.split('.')[1:])]
        else:
            model_state_dict[name] = state_dict[name]
    model.load_state_dict(model_state_dict)
    return model
    
if  __name__ == '__main__':
    
    pretrained_weights = torch.load('./hand/pose_iter_102000.caffemodel.pt')
    model = HandPoseModel()

    model = load_weights(model,pretrained_weights)
    print(model.prev_stage6.Mconv7_stage6.bias)
    #print(.conv2_1.weight.shape)
    print(pretrained_weights['Mconv7_stage6.bias'])
    

    image = cv2.cvtColor(cv2.imread(TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (960,540))
    image_out = process(model, image)
    plt.figure(figsize=(12,12))
    plt.imshow(image_out)
    plt.imsave('image2.png',image_out)
    plt.show()
    
    """
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while 1:
        ret, image = cap.read()
        clear_output()

        image_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #start = clock()
        image_out = process(model, image)
        #stop = clock()
        #took = stop-start
        #cv2.putText(image_out,'Inference: {}s, post: {}s'.format(  np.round(inference_took,3) , np.round(took-inference_took,3) ),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)

    #    cv2.imshow("OpenPose's stolen hand tracking network in Keras", cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))
        cv2.imshow("OpenPose's stolen hand tracking network in Keras", image_out)
        cv2.waitKey(1)

    cap.release()
    """
