import torch
import numpy as np
import torch.nn as nn


def g(x, alpha, mu, sigma1, sigma2):
    # Calculate sigma and apply np.clip
    sigma = np.clip((x < mu) * sigma1 + (x >= mu) * sigma2, a_min=1e-6, a_max=None)
    return alpha * np.exp((x - mu)**2 / (-2 * (sigma**2)))


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

class ColourSystem(nn.Module):
    def __init__(self, bands, cs='sRGB', device='cuda'):
        super().__init__()

        #rgb_wavelengths = [620, 555, 503]

        #nearest = np.array(bands).reshape(-1, 1) - np.array(rgb_wavelengths).reshape(1, -1)
        #self.nearest = torch.from_numpy(np.argmin(np.abs(nearest), axis=0)).to(device)

        #white = np.array([176.90352, 94.22424, 101.18808]) / 255.0
        #self.white = torch.from_numpy(white).to(device).float()

        bands = np.array(bands) * 10
        cmf = np.array([component_x(bands), component_y(bands), component_z(bands)])

        red, green, blue, white = COLOR_SPACE[cs]
        M = np.vstack((red, green, blue)).T
        MI = np.linalg.inv(M)
        wscale = MI.dot(white)
        A = MI / wscale[:, np.newaxis]

        XYZ = cmf
        RGB = XYZ.T @ A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)

        # Register buffer instead of attribute
        self.register_buffer(
            'transform_matrix',
            torch.from_numpy(RGB).float()
        )
    
        #white = np.array([156.90352, 104.22424, 101.18808]) / 255.0
        #white = torch.from_numpy(white).float()

        #self.register_buffer(
        #    'transform_matrix',
        # torch.from_numpy(RGB).float() * white
        #)


        #self.transform_matrix_learnable = nn.Parameter(torch.rand_like(self.transform_matrix), requires_grad=True)

    def gamma_correction(self, x):
        result = torch.where(
            x < 0.0031308,
            12.92 * x,
            1.055 * (x.clamp(min=1e-6).pow(1 / 2.4)) - 0.055
        )

        return result

    @torch.cuda.amp.autocast()
    def forward(self, spec):

        rgb = torch.matmul(spec, self.transform_matrix)

        rgb = self.gamma_correction(rgb)
        #rgb = spec[:,self.nearest] / self.white
        #rgb = torch.pow(rgb, 1/1.7)

        #rgb = rgb / 1.8

        #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        rgb = rgb.clamp(0, 1)

        return rgb