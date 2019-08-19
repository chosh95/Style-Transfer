"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from smooth_filter import smooth_filter


class ReMapping:
    def __init__(self):
        self.remapping = []

    def process(self, seg):
        new_seg = seg.copy()
        for k, v in self.remapping.items():
            new_seg[seg == k] = v
        return new_seg


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def memory_limit_image_resize(con_img):
    # prevent too small or too big images
    MINSIZE=256
    MAXSIZE=960
    orig_width = con_img.width
    orig_height = con_img.height
    if max(con_img.width,con_img.height) < MINSIZE:
        if con_img.width > con_img.height:
            con_img.thumbnail((int(con_img.width*(1.0/con_img.height)*MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            con_img.thumbnail((MINSIZE, int(con_img.height*1.0/con_img.width*MINSIZE)), Image.BICUBIC)
    if min(con_img.width,con_img.height) > MAXSIZE:
        if con_img.width > con_img.height:
            con_img.thumbnail((MAXSIZE, int(con_img.height*1.0/con_img.width*MAXSIZE)), Image.BICUBIC)
        else:
            con_img.thumbnail(((int(con_img.width*1.0/con_img.height*MAXSIZE), MAXSIZE)), Image.BICUBIC)
    print("Resize image: (%d,%d)->(%d,%d)" % (orig_width, orig_height, con_img.width, con_img.height))
    return con_img.width, con_img.height


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cuda, save_intermediate, no_post, cont_seg_remapping=None, styl_seg_remapping=None):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')
        
        '''
        new_cont = cont_img.crop((120,120,480,600))
        new_cont.save("cont.jpg")
        new_styl = styl_img.crop((120,50,480,650))
        new_styl.save("styl.jpg")
        '''

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)

        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)


        if cuda:
            cont_img = cont_img.cuda(0)
            styl_img = styl_img.cuda(0)
            stylization_module.cuda(0)

        # cont_img = Variable(cont_img, volatile=True)
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)

##########################
        cont = cont_img.squeeze(0)
        max_label = np.max(cont_seg) + 1 #label 개수
        label_set = np.unique(cont_seg) #[0,1,2,..]
        cont_c, cont_h, cont_w = cont.size(0), cont.size(1), cont.size(2)      # 3,h,w  

        result = Image.open('result.jpg').convert('RGB')
        result = transforms.ToTensor()(result)
        result = result.cuda(0)
        print(result.shape)
        result = result.view(3,-1).clone()
        print(result.shape)
        '''   
        for i in range(0,cont_c):
            for j in range(0,cont_h):
                for k in range(0,cont_w):
                    if cont_seg[j][k]==0:
                        cont[i][j][k] = 255
        print(cont.shape)   

        cont = Image.open(content_image_path).convert('RGB')
        cont = np.asarray(cont)
        cont = np.transpose(cont,(1,2,0))
        img = Image.fromarray(cont,'RGB')
        img.save('cont.jpg') 
        '''
        cont_view = cont.view(cont_c, -1).clone() # 3차원을 2차원으로 조정 3 x w*h
         
       
        for l in label_set:
            cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l) #l과 label값이 같은 곳의 위치 저장하는 배열
            cont_indi = torch.LongTensor(cont_mask[0]) #마스크를 long으로 자료형 변환 #(1,z<=x*y)
            cont_indi = cont_indi.cuda(0)
            cont = torch.index_select(cont_view, 1, cont_indi)
           
            if l == 1 :    
                new_cont = torch.transpose(cont_view,1,0)
                new_cont.index_copy_(0,cont_indi,torch.transpose(result,1,0))
                cont_view = torch.transpose(new_cont,1,0)

        print(cont_view.shape)
        cont_view = torch.reshape(cont_view,(761,596,3))
        cont_view = np.asarray(cont_view)
        print(cont_view.shape)
        im = Image.fromarray(cont_view,'RGB')
        im.save("output.jpg")
        


        cont = torch.reshape(cont,(3,207,937))
        cont = np.asarray(cont)
        np.save('cont',cont)
        print(cont.shape)
       
        
        

##########################
        '''
        if cont_seg_remapping is not None:
            cont_seg = cont_seg_remapping.process(cont_seg)
        if styl_seg_remapping is not None:
            styl_seg = styl_seg_remapping.process(styl_seg)
        
        if save_intermediate:
            with Timer("Elapsed time in stylization: %f"):
                stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
            if ch != new_ch or cw != new_cw:
                print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
            utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1, padding=0)

            with Timer("Elapsed time in propagation: %f"):
                out_img = smoothing_module.process(output_image_path, content_image_path)
            out_img.save(output_image_path)

            if not cuda:
                print("NotImplemented: The CPU version of smooth filter has not been implemented currently.")
                return

            if no_post is False:
                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter(output_image_path, content_image_path, f_radius=15, f_edge=1e-1)
            out_img.save(output_image_path)
        else:
            with Timer("Elapsed time in stylization: %f"):
                stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
            if ch != new_ch or cw != new_cw:
                print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
            grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out_img = Image.fromarray(ndarr)

            grid4 = utils.make_grid(cFFG.data, nrow=1, padding=0)
            ndarr4 = grid4.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out_img4 = Image.fromarray(ndarr4)
            out_img4.save('./results/cFFG.png')
            
            with Timer("Elapsed time in propagation: %f"):
                out_img = smoothing_module.process(out_img, cont_pilimg)

            if no_post is False:
                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
            out_img.save(output_image_path)
            '''

