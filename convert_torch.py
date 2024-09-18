import torch


def linearRGB_from_sRGB(im: torch.tensor) -> torch.tensor:
    # Convert sRGB to linearRGB (copied from daltonlens.convert.sRGB_from_linearRGB)
    out = torch.empty_like(im)
    small_mask = im < 0.04045
    large_mask = torch.logical_not(small_mask)
    out[small_mask] = im[small_mask] / 12.92
    out[large_mask] = torch.pow((im[large_mask] + 0.055) / 1.055, 2.4)
    return out


def sRGB_from_linearRGB(im: torch.tensor) -> torch.tensor:
    # Convert linearRGB to sRGB. Made on the basis of daltonlens.convert.sRGB_from_linearRGB
    # by Nicolas Burrus. Clipping operation was removed.
    out = torch.empty_like(im)
    small_mask = im < 0.0031308
    large_mask = torch.logical_not(small_mask)
    out[small_mask] = im[small_mask] * 12.92
    out[large_mask] = torch.pow(im[large_mask], 1.0 / 2.4) * 1.055 - 0.055
    return out


def xyz_to_lab(img):
    '''
    Convert from XYZ to LAB color space according to D65
    '''
    def lab_func(x):
        mask1 = x > (6/29)**3
        mask2 = x <= (6/29)**3
        result = torch.zeros_like(x)
        result[mask1] = x[mask1]**(1/3)

        result[mask2] = 1/3*(29/6)**2*x[mask2] + 4/29
        
        return result
    
    # D65 white point
    Xn, Yn, Zn = (0.9504, 1.000, 1.0888)
        
    L = 116.*lab_func(img[..., 1]/Yn)-16.
    a = 500.*(lab_func(img[..., 0]/Xn)-lab_func(img[..., 1]/Yn))
    b = 200.*(lab_func(img[..., 1]/Yn)-lab_func(img[..., 2]/Zn))
    
    return torch.stack([L, a, b], dim=2)


def sRGB_to_lab(img, device='cpu'):
    '''
    Convert from sRGB to LAB color space according to D65
    '''
    # wikipedia
    # XYZ_from_RGB = torch.tensor([
    #                [0.49000, 0.31000, 0.20000],
    #                [0.17697, 0.81240, 0.01063],
    #                [0.00000, 0.01000, 0.99000]
    # ])
    
    # From sRGB specification
    xyz_from_rgb= torch.FloatTensor([[0.412453, 0.357580, 0.180423],
                                [0.212671, 0.715160, 0.072169],
                                [0.019334, 0.119193, 0.950227]]).to(device)
    
    linRGB = linearRGB_from_sRGB(img)
    xyz = linRGB@xyz_from_rgb.T
    lab = xyz_to_lab(xyz)

    return lab
