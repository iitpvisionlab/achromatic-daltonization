import os
# import sys
# sys.path.insert(0, '../common_modules/')

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from convert_torch import linearRGB_from_sRGB, sRGB_from_linearRGB


def collate_fn(batch):
    new_batch = []
    names = []
    for image, path, h_max, w_max in batch:
        pad_h = h_max-image.shape[1]
        pad_w = w_max-image.shape[2]
        tmp_image = F.pad(image, (0, pad_w, 0, pad_h), 'constant')
        new_batch.append(tmp_image[None])
        names.append(path)

    return torch.cat(new_batch, axis=0), names


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, h_max, w_max, train=True):
        self.image_paths = image_paths
        self.h_max = h_max
        self.w_max = w_max

    def transform(self, image):  
        # Transform to tensor
        image = image
        image = TF.to_tensor(image).to(torch.float32)
        
        return image

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        x = self.transform(image)
        _, tail = os.path.split(self.image_paths[index])
        return x, tail.split('.')[0], self.h_max, self.w_max

    def __len__(self):
        return len(self.image_paths)

def simulation_func(device, sim_func, cvdtype):
    if sim_func == 'Farup':
        if cvdtype == 'rg':
            mat = torch.tensor([[.5, .5, 0],
                                [.5, .5, 0],
                                [0, 0, 1]]).to(device)
        elif cvdtype == 'by':
            mat = torch.tensor([[.75, -.25, .5],
                                [-.25, .75, .5],
                                [.25, .25, .5]]).to(device)
                                
        return lambda x: x@mat.T
    elif sim_func == 'Vienot':
        LMS_from_RGB = torch.tensor([
            [ 0.27293945,  0.66418685,  0.06287371],
            [ 0.10022701,  0.78761123,  0.11216177],
            [ 0.01781695,  0.10961952,  0.87256353]]
        ).to(device)
    
        RGB_from_LMS = torch.tensor([
            [ 5.30329968, -4.49954803,  0.19624834],
            [-0.67146001,  1.86248629, -0.19102629],
            [-0.0239335 , -0.14210614,  1.16603964]]
        ).to(device)
        if cvdtype == 'PROTAN':
            # protan sim matrix
            sim_matrix = torch.tensor([
                [ 0.        ,  1.06481845, -0.06481845],
                [ 0.        ,  1.        ,  0.        ],
                [ 0.        ,  0.        ,  1.        ]]
            ).to(device)
        if cvdtype == 'DEUTAN':
            # deutan sim matrix
            sim_matrix = torch.tensor([
                [ 1.        ,  0.        ,  0.        ],
                [ 0.93912723,  0.        ,  0.06087277],
                [ 0.        ,  0.        ,  1.        ]]
            ).to(device)
        def Vienot_sim(img):
            linRGB = linearRGB_from_sRGB(img)
            LMS = linRGB@LMS_from_RGB.T
            sim_LMS = LMS@sim_matrix.T
            inv_linRGB = sim_LMS@RGB_from_LMS.T
            
            inv_sRGB = sRGB_from_linearRGB(inv_linRGB)
            return inv_sRGB

        return Vienot_sim

def get_dw(batch, device, sim_func, sign_guide, px, dim, method, thresh, av_v_fill=0.8):
    '''
    Find direction and its module
    Solve simple quadratic equation

    Arguments:
    batch: torch.tensor, [N, C, H, W]
        batch of images
        N -- batch size, C -- channels,
        H -- image height, W -- image width
    device: str, ['cpu', 'cuda:{i}']
        tensor device
    sim_func: function(image)
        simulation function
    sign_guide: torch.tensor, [N, H, W]
        refer to get_sign_guide function
    px: int,
        bias of dim to get discrete derivatives
    dim: int,
        dim for derivatives computation
    method: str, ['approxMulitplicat']

    thresh: float
        threshold for sign guide
        currently have just one option -- 0
    av_w_fill: float
        assumption of watermark average value
    '''
    # N, H, W, 3
    perm_batch = batch.permute(0, 2, 3, 1)
    dc = perm_batch-torch.roll(perm_batch, shifts=px, dims=dim)
        
    b, c, h, w = batch.shape
    
    if method=='approxMultiplicat':
        # N, H, W, 1
        av_w_3d = av_v_fill*torch.ones(b, h, w, 1).to(device)

        # N, H, W, 3
        a = sim_func((perm_batch +
                      torch.roll(perm_batch, 
                                 shifts=px, 
                                 dims=dim))/2)
        # N, H, W, 3
        b = av_w_3d*(sim_func(dc))

    # N, H, W
    c = torch.sum(dc**2, dim=3)
    
    D_pt2 = 4*torch.sum(a**2, dim=3) * (torch.sum(b**2, dim=3)-c)
    
    D_pt1 = 2*torch.sum(a*b, dim=3)
    
    # N, H, W
    D = D_pt1**2 - D_pt2
    
    print(f'D < 0: {(D < 0).sum()}')
    print(f'DMax: {(D[D <= 0]).min()}')
    print(D[D < 0])
    # crutch
    D[D < 0] = 0

    dw1 = (-D_pt1 + torch.sqrt(D))/(2*torch.sum(a**2, dim=3))
    dw2 = (-D_pt1 - torch.sqrt(D))/(2*torch.sum(a**2, dim=3))

    sign = sign_guide - torch.roll(sign_guide, shifts=px, dims=dim)
    mask_lesser_dw = sign < -thresh
    mask_larger_dw = sign > thresh

    dw_larger = torch.where(dw1 >= dw2, dw1, dw2)
    dw_lesser = torch.where(dw1 <= dw2, dw1, dw2)

    out = torch.zeros_like(dw1)
    out[mask_larger_dw] = dw_larger[mask_larger_dw]
    out[mask_lesser_dw] = dw_lesser[mask_lesser_dw]

    return out


def get_sign_guide(img_torch, device, sign_g, sim_func_name, CVD_type):
    '''
    Getting sign guide for daltonization 
    approximation problem (refer to get_dw)

    Arguments:
    img_torch: torch.tensor, [N, C, H, W]
        N -- batch size, C -- channels,
        H -- image height, W -- image width
    device: str, ['cpu', 'cuda:{i}']
        tensor device
    sign_g: str, ['l', 'b', 'lb', 'lbb', 's_l']
        l -- stands for luminance channel
        b -- stands for blind channel
        lb -- combination of luminance and blind channels
        sign_guide
        lbb -- also combination
        s_l -- stands for luminance of simulation (now working in Vienot simulations)
    sim_func_name: str, ['Farup_simple', 'Vienot']
        variant of simulation, for Vienot it works in linRGB space
    CVD_type: str, ['rg', 'by', 'PROTAN', 'DEUTAN']
        'rg' -- red-green Farup dichromate
        'by' -- blue-yellow Farup dichromate
        'PROTAN' -- real Protanope
        'DEUTAN' -- real Deuteranope
    '''
    blind_3d = img_torch.permute(0, 2, 3, 1)
    assert sign_g in ['lb', 'lbb', 'b', 'l', 's_l'], "no such option for sign guide"

    if sim_func_name == 'Farup':
        if CVD_type=='rg':
            if sign_g=='lb':
                vec4blnd = [2, 0, 1]
                sign_guide = ((blind_3d[..., 0]*vec4blnd[0] + 
                               blind_3d[..., 1]*vec4blnd[1] + 
                               blind_3d[..., 2]*vec4blnd[2])/torch.tensor(3))
            elif sign_g=='lbb':
                vec4blnd = [1.5, -0.5, 0.5]
                sign_guide = ((blind_3d[..., 0]*vec4blnd[0] + 
                               blind_3d[..., 1]*vec4blnd[1] + 
                               blind_3d[..., 2]*vec4blnd[2])/torch.tensor(3))
            elif sign_g=='b':
                vec4blnd = [1, -1, 0]
                sign_guide = ((blind_3d[..., 0]*vec4blnd[0] + 
                               blind_3d[..., 1]*vec4blnd[1] + 
                               blind_3d[..., 2]*vec4blnd[2])/torch.tensor(3))
        elif CVD_type=='by':
            if sign_g=='b':
                vec4blnd = [-1, -1, 1]
                blind_1d = (blind_3d[..., 0]*vec4blnd[0] + 
                            blind_3d[..., 1]*vec4blnd[1] + 
                            blind_3d[..., 2]*vec4blnd[2])/torch.tensor(3)
                sign_guide = torch.tensor(blind_1d+torch.tensor(torch.empty(img_torch.shape[1], img_torch.shape[2]).fill_(2/3)))
    elif sim_func_name == 'Vienot':
        # get blind channel
        # img_torch should be in linRGB space
        LMS_from_RGB = torch.tensor([[ 0.27293945,  0.66418685,  0.06287371],
                                     [ 0.10022701,  0.78761123,  0.11216177],
                                     [ 0.01781695,  0.10961952,  0.87256353]]).to(device)
        
        RGB_from_LMS = torch.tensor([[ 5.30329968, -4.49954803,  0.19624834],
                                     [-0.67146001,  1.86248629, -0.19102629],
                                     [-0.0239335 , -0.14210614,  1.16603964]]).to(device)
        protan_Vienot = torch.tensor([[ 0.        ,  1.06481845, -0.06481845],
                                      [ 0.        ,  1.        ,  0.        ],
                                      [ 0.        ,  0.        ,  1.        ]]).to(device)
        deutan_Vienot = torch.tensor([[ 1.        ,  0.        ,  0.        ],
                                     [ 0.93912723,  0.        ,  0.06087277],
                                     [ 0.        ,  0.        ,  1.        ]]).to(device)
                                                                                                    
        if CVD_type == 'PROTAN':
            # here blind_3d should be in linRGB
            if sign_g == 'b':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                I = torch.FloatTensor([[1, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]).to(device)
                blind_LMS= LMS@I.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                blind_sum = blind_sRGB.mean(axis=3)
                
                return blind_sum
            elif sign_g == 'l':
                return blind_3d.mean(axis=3)
            elif sign_g == 'lb':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                I = torch.FloatTensor([[1, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]).to(device)
                blind_LMS= LMS@I.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                blind_sum = blind_sRGB.mean(axis=3)
                
                return (blind_sum + blind_3d.mean(axis=3))/2
            elif sign_g == 's_l':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                
                blind_LMS = LMS@protan_Vienot.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                
                return blind_sRGB.mean(axis=3)
            
        if CVD_type == 'DEUTAN':
            # here blind_3d should be in linRGB
            if sign_g == 'b':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                I = torch.FloatTensor([[0, 1, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]).to(device)
                blind_LMS= LMS@I.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                blind_sum = blind_sRGB.mean(axis=3)
                
                return blind_sum
            elif sign_g == 'l':
                return blind_3d.mean(axis=3)
            elif sign_g == 'lb':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                I = torch.FloatTensor([[0, 1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]).to(device)
                blind_LMS= LMS@I.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                blind_sum = blind_sRGB.mean(axis=3)
                
                return (blind_sum + blind_3d.mean(axis=3))/2
            elif sign_g == 's_l':
                linRGB = linearRGB_from_sRGB(blind_3d)
                LMS = linRGB@LMS_from_RGB.T
                
                blind_LMS = LMS@deutan_Vienot.T
                blind_linRGB = blind_LMS@RGB_from_LMS.T
                blind_sRGB = sRGB_from_linearRGB(blind_linRGB)
                
                return blind_sRGB.mean(axis=3)
    return sign_guide


def M2(x, dw_y, dw_x, px, epsilon):    
    x_x = x - torch.roll(x, shifts=px, dims=2)
    x_y = x - torch.roll(x, shifts=px, dims=1)

    matErr = (x_x-dw_x)**2/(dw_x**2 + epsilon**2)+(x_y-dw_y)**2/(dw_y**2 + epsilon**2)

    return matErr.mean(axis=(-1, -2))


def M1_diff(tsr, px, dim):
    return tsr - torch.roll(tsr, shifts=px, dims=dim)


def M1(x, px, img_torch, inNorm2_x, inNorm2_y, sim_func, epsilon):
    simulated = sim_func(img_torch.permute(0, 2, 3, 1)*x.unsqueeze(-1))

    grad_x_simulated = M1_diff(simulated, px, 2)
    grad_y_simulated = M1_diff(simulated, px, 1)

    simNorm2_x = torch.sum(grad_x_simulated**2, dim=3)
    simNorm2_y = torch.sum(grad_y_simulated**2, dim=3)

    matErr = (((simNorm2_x+1E-32)**(1/2)-(inNorm2_x+1E-32)**(1/2))**2/(inNorm2_x + epsilon**2) + 
             ((simNorm2_y+1E-32)**(1/2)-(inNorm2_y+1E-32)**(1/2))**2/(inNorm2_y + epsilon**2))

    matErr = matErr.mean(axis=(1, 2))
    return matErr
