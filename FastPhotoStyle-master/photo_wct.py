"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.utils as utils
from models import VGGEncoder, VGGDecoder


class PhotoWCT(nn.Module):
    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
    
    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)

        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0) #(1,channel,weight,height) -> (channel,weight,height)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg,0)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg,0)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg,0)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(cont_img)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg,0)
        Im1 = self.d1(csF1)

        '''
        c4,cp,cp1,cp2,cpl2,cp3,cpl3 = self.e4(cont_img)
        s4,sp,sp1,sp2,spl2,sp3,spl3 = self.e4(styl_img)
        f4 = (s4 + c4) / 2
        cF4 = self.d4(f4,cp,cp1,cp2,cpl2,cp3,cpl3)
        sF4 = self.d4(f4,sp,sp1,sp2,spl2,sp3,spl3)

        c3,cp,cp1,cp2,cpl2 = self.e3(cF4)
        s3,sp,sp1,sp2,spl2 = self.e3(sF4)
        f3 = (s3 + c3) / 2
        cF3 = self.d3(f3,cp,cp1,cp2,cpl2)
        sF3 = self.d3(f3,sp,sp1,sp2,spl2)

        c2,cp,cpl = self.e2(cF3)
        s2,sp,spl = self.e2(sF3)
        f2 = (s2 + c2) / 2        
        cF2 = self.d2(f2,cp,cpl)
        sF2 = self.d2(f2,sp,spl)


        c1 = self.e1(cF2)
        s1 = self.e1(sF2)
        f1 = (s1 + c1) / 2
        cF1 = self.d1(f1)
        sF1 = self.d1(f1)
        '''
        return Im1,Im1,Im1,Im1, Im1

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1 #label 개수
        self.label_set = np.unique(cont_seg) #[0,1,2,..]
        self.label_indicator = np.zeros(max_label) #[0,0,0,0,..]
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
	    #레이블의 위치 저장
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
	    #조건 부합하는지에 따라 1.0, 0.0 저장

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg, cont_feat_rate):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        cont_feat_view = cont_feat.view(cont_c, -1).clone() # 3차원을 2차원으로 조정
        styl_feat_view = styl_feat.view(styl_c, -1).clone() # ex. (512,x,y) -> (512,x*y)
        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__wct_core(cont_feat_view, styl_feat_view)
        else:
#            print(cont_feat)
            target_feature = cont_feat.view(cont_c, -1).clone() # 512 * z 
#            print(target_feature)
            if len(cont_seg.shape) == 2: # 2차원인 경우 
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST)) #content segment image를 feature의 width, height크기에 맞춰서 조정 후 numpy로 전환해서 저장
            else: #그 이상인 경우
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:  #style segment도 마찬가지로 크기조정
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))
            
            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue #indicator[l]이 false면 즉 변환할 필요가 없는 레이블이면 continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l) #l과 label값이 같은 곳의 위치 저장하는 배열
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l) # z = x*y 중 label이 같은 일부
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0]) #마스크를 long으로 자료형 변환 #(1,z<=x*y)
                styl_indi = torch.LongTensor(styl_mask[0])
                if self.is_cuda:
                    cont_indi = cont_indi.cuda(0)
                    styl_indi = styl_indi.cuda(0)

                cFFG = torch.index_select(cont_feat_view, 1, cont_indi) #마스크와 인덱스가 같은 픽셀들 선택 c * z
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi) 
                # print(len(cont_indi)) 
                # print(len(styl_indi))
                tmp_target_feature = self.__wct_core(cFFG, sFFG) #실질적인 전환
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    tmp_target_feature = cont_feat_rate*cFFG + (1-cont_feat_rate)*tmp_target_feature # 원본 : 변환 비율에 맞춰서 저장##############################################################
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, \
                            		torch.transpose(tmp_target_feature,1,0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat) # c x (hxw) -> c x h x w 
        ccsF = target_feature.float().unsqueeze(0) # c x h x w  -> 1 x c x h x w
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size() # c x z     (ex. 512 * 7214)
        c_mean = torch.mean(cont_feat, 1)  # 행의 평균을 구한 1차원 배열 [c, ] 
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat) # 1차원 배열을 다시 2차원 c x z로 확장
        # [1,2,3]이었으면 [[1,1,1],[2,2,2],[3,3,3]]으로, 즉 계산을 위한 차원 확장
        cont_feat = cont_feat - c_mean # feature에 행(c)의 평균값 제거
        
        iden = torch.eye(cFSize[0])  # .double()
        # c크기대로 대각선만 1.0 인 행렬 [[1,0,0],[0,1,0],[0,0,1]], 즉 단위행렬

        if self.is_cuda:
            iden = iden.cuda()
        
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden # 고유벡터와 고유값을 구하기 위한 과정
        # (CFFG * CFFG.T) / (h*w -1)    +   iden
	# 크기는 c x c 
        # del iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break

	#style image도 content와 같은 과정 계산
        sFSize = styl_feat.size() # c x (hxw)
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)


        epsilon = 0.001 # 0으로 나뉘는걸 방지하는 hyper parameter
        ep_mat = torch.ones(k_c,dtype=torch.float32,device='cuda') * epsilon
        c_d = c_d + ep_mat

        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t())) #P_C = c_v * 1/root(c_d) * c_v.T
        whiten_cF = torch.mm(step2, cont_feat) #P_c * H_c

        s_d = (s_e[0:k_s]).pow(0.5)

        epsilon2 = 0.001 # 0으로 나뉘는걸 방지하는 hyper parameter
        ep_mat2 = torch.ones(k_s,dtype=torch.float32,device='cuda') * epsilon2
        s_d = s_d + ep_mat2

        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF) # P_S = s_c * root(s_v) * s_c.T
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature) #featue = P_s * P_c * H_c
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass
