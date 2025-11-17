import os
import numpy as np
import pandas as pd
import torch
import cv2
import OpenEXR
import Imath
import argparse
import numpy as np


def g(x, alpha, mu, sigma1, sigma2):
    sigma = (x < mu)*sigma1 + (x >= mu)*sigma2
    return alpha*np.exp((x-mu)**2 / (-2*(sigma**2)))


def component_x(x): return g(x, 1.056, 5998, 379, 310) + \
    g(x, 0.362, 4420, 160, 267) + g(x, -0.065, 5011, 204, 262)


def component_y(x): return g(x, 0.821, 5688, 469, 405) + \
    g(x, 0.286, 5309, 163, 311)


def component_z(x): return g(x, 1.217, 4370, 118, 360) + \
    g(x, 0.681, 4590, 260, 138)


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E':  xyz_from_xy(1/3, 1/3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


class ColourSystem:

    def __init__(self, start=450, end=650, num=21, cs='sRGB'):

        bands = np.linspace(start=start, stop=end, num=num)*10

        self.cmf = np.array([component_x(bands),
                             component_y(bands),
                             component_z(bands)])

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cs]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

  
    def get_transform_matrix(self):

        XYZ = self.cmf
        RGB = XYZ.T @ self.A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)
        return RGB

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        M = self.get_transform_matrix()
        rgb = spec @ M
        return rgb



def read_exr_as_np(fn):
   f = OpenEXR.InputFile(fn)
   channels = f.header()['channels']
   dw = f.header()['dataWindow']
   size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
   ch_names = []
   image = np.zeros((size[1], size[0], len(channels)))
   for i, ch_name in enumerate(channels):
       ch_names.append(ch_name)
       ch_dtype = channels[ch_name].type
       ch_str = f.channel(ch_name, ch_dtype)
      
       if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
           np_dtype = np.float32
       elif ch_dtype == Imath.PixelType(Imath.PixelType.HALF):
           np_dtype = np.half
          
       image_ch = np.fromstring(ch_str, dtype=np_dtype)
       image_ch.shape = (size[1], size[0])
       image[:,:,i] = image_ch
  
   return image, ch_names


def color_mapping(PATH, OBJECT, wavelength, FILE, split, scalar=1, s123_min=-1, s123_max=1):
   img_path = f"{PATH}/{OBJECT}/{wavelength}/{split}/{FILE}"
   image, _ = read_exr_as_np(img_path)
   print(img_path, image.shape)
   stokes = np.clip(image, -1, 1)
   s = stokes[..., 0]
   return s


def spec_to_rgb(img_path, save_path, view, split):
    wvls = np.arange(450, 650+1, 10)
    cube = np.zeros((512, 512, len(wvls)))
    
    for i, wvl in enumerate(wvls):
        s = color_mapping(".", img_path, wvl, f"{view}.exr", split)
        cube[..., i] = s
    
    os.makedirs(save_path, exist_ok=True)
    cube = cube.clip(0, 1)
    np.save(os.path.join(save_path, f"{view}.npy"), cube)

    spectorgb = ColourSystem(cs='sRGB')
    rgb = spectorgb.spec_to_rgb(cube.reshape(-1, cube.shape[-1]))
    
    rgb = rgb.clip(0, 1)
    gamma_correct = lambda x: np.where(x < 0.0031308, 12.92*x, 1.055*(x**(1.0/2.4))-0.055)
    rgb = gamma_correct(rgb)
    
    imgs_rgb = rgb.reshape(512, 512, 3)[...,::-1] * 255
    cv2.imwrite(os.path.join(save_path, f"{view}.png"), imgs_rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--scene', type=str, required=True, help='Scene name (e.g. ajar)')
    args = parser.parse_args()
    
    if args.scene == "ajar" or args.scene == "cbox_sphere" or args.scene == "cbox_dragon":
        idxs = list(range(43))
        if args.split == "train":
            remove_idx = [0,8,16,24,32,40,48]  #review idxs
            idxs = [i for i in idxs if i not in remove_idx]
        else:  # val
            idxs = [0,8,16,24,32,40,48]
    else:
        idxs = list(range(200)) # review hotdog val idxs

    if args.split == 'test' or args.split == 'val':
        savefolder = 'eval'
    else:
        savefolder = args.split

    os.makedirs(f"./processed/{args.scene}/{savefolder}/", exist_ok=True)

    for k in idxs:
        spec_to_rgb(f"{args.scene}", f"./processed/{args.scene}/{savefolder}/",
                    "r_{}".format(k), args.split)


if __name__ == "__main__":
   main()