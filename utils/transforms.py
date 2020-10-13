import collections
import random
import matplotlib.tri as mtri
import cv2
import numpy as np
import scipy.sparse
import torch
from PIL import Image, ImageFilter, ImageOps

class CreateNewItem(object):
    def __init__(self, transforms, key, new_key):
        self.transforms = transforms
        self.key = key
        self.new_key = new_key

    def __call__(self, input_dict):
        input_dict[self.new_key] = self.transforms(input_dict[self.key])
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key) + ', ' + str(self.new_key)
        format_string += ')'
        return format_string



class GenerateTranslatedKP(object):
    def __init__(self, anchor_pts):
        self.anchor_pts = anchor_pts

    def generate_offset_map(self, source, target):
        anchor_pts = [[0, 0], [0, 256], [256, 0], [256, 256],
                      [0, 128], [128, 0], [256, 128], [128, 256],
                      [0, 64], [0, 192], [256, 64], [256, 192],
                      [64, 0], [192, 0], [64, 256], [192, 256]]
        anchor_pts = np.asarray(anchor_pts) / 256
        # print(f"shape of spoof: {source.shape}, shape of live: {target.shape}")
        # print(f"shape of anch: {anchor_pts.shape}")
        xi, yi = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        # print(f"shape of xi: {xi.shape}, shape of yi: {yi.shape}")
        _source = np.concatenate([source, anchor_pts], axis=0).astype(np.float32)
        _target = np.concatenate([target, anchor_pts], axis=0).astype(np.float32)
        # print(f"shape of spoof after meshgrid: {_source.shape}, shape of live: {_target.shape}")
        _offset = _source - _target
        # print(f"shape of offset: {_offset.shape}")
        # interp2d
        _triang = mtri.Triangulation(_target[:, 0], _target[:, 1])
        # print(f"after triangulation: {_triang}")
        _interpx = mtri.LinearTriInterpolator(_triang, _offset[:, 0])
        _interpy = mtri.LinearTriInterpolator(_triang, _offset[:, 1])
        # print(f"shapes interps: {_interpx}, {_interpy}")
        _offsetmapx = _interpx(xi, yi)
        _offsetmapy = _interpy(xi, yi)
        # print(f"x, y offset: {_offsetmapx.shape}, {_offsetmapy.shape}")
        offsetmap = np.stack([_offsetmapy, _offsetmapx, _offsetmapx * 0], axis=2)
        # print(f"final offsetmap: {offsetmap.shape}")
        return offsetmap

    def __call__(self, spoof, live):
        return self.generate_offset_map(spoof, live)



class ConvertColor(object):
    def __init__(self, color_mode):
        self.color_mode = color_mode

    def __call__(self, img):
        img = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{self.color_mode.upper()}"))
        return img


class LoadNP(object):

    def __call__(self, img_path):
        array = np.load(img_path)
        # print(array.shape)
        return array

class JointTransforms(object):
    def __init__(self, transforms, key_list):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        input_list = [item for x in self.key_list for item in input_dict[x]]
        # print(len(input_list), input_list[0].shape)")
        # print("joint transforms ", self.transforms)
        for t in self.transforms:
            # print(t)
            input_list = t(input_list)

        for idx, key in enumerate(self.key_list):
            input_dict[key] = [input_list[idx]]
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key_list)
        format_string += ')'
        return format_string

class Transform4EachLabel(object):
    """
    Applies transforms only to chosen labels
    """

    def __init__(self, transforms, label='label', allowed_labels=[0, 1]):
        self.label = label
        self.allowed_labels = allowed_labels if type(allowed_labels) == list else [allowed_labels]
        self.transforms = transforms

    def __call__(self, input_dict):
        dict_label = input_dict[self.label]
        # print(dict_label, self.allowed_labels, set(self.allowed_labels))

        if dict_label in self.allowed_labels:
            return self.transforms(input_dict)
        else:
            return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.label)
        format_string += '\n)'
        return format_string


class Transform4EachKey(object):
    """
    Apply all torchvision transforms to dict by each key
    """

    def __init__(self, transforms, key_list):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        for key in self.key_list:
            # print(self.transforms)
            for t in self.transforms:
                input_dict[key] = t(input_dict[key])

        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key_list)
        format_string += '\n)'
        return format_string


class Transform4EachElement(object):
    """
    Apply all transforms to list for each element
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_list):

        if isinstance(input_list, Image.Image):

            for t in self.transforms:

                input_list = t(input_list)
        else:
            for idx in range(len(input_list)):
                # print("fdsfsdf", type(input_list))

                for t in self.transforms:

                    input_list[idx] = t(input_list[idx])
        return input_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += '\n)'
        return format_string



class StackTensors(object):
    """
    Stack list of tensors to one tensor
    """

    def __init__(self, squeeze=False):
        self.squeeze = squeeze

    def __call__(self, input_list):
        res_tensor = torch.stack(input_list)
        if self.squeeze:
            res_tensor = res_tensor.squeeze()
        return res_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'({self.squeeze})'



class GaussianBlur(object):
    """
    Apply Gaussian blur to image with probability 0.5
    """

    def __init__(self, max_blur_kernel_radius=3, rand_prob=0.5):
        self.max_radius = max_blur_kernel_radius
        self.rand_prob = rand_prob

    def __call__(self, img):
        radius = np.random.uniform(0, self.max_radius)
        if np.random.random() < self.rand_prob:
            return img.filter(ImageFilter.GaussianBlur(radius))
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.max_radius)


class GaussianNoise(object):
    """
    Apply Gaussian noise to image with probability 0.5
    """

    def __init__(self, var_limit=(10.0, 50.0), mean=0., rand_prob=0.5):
        self.var_limit = var_limit
        self.mean = mean
        self.rand_prob = rand_prob

    def __call__(self, img):
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        np_img = np.array(img)
        gauss = np.random.normal(self.mean, sigma, np_img.shape)
        if np.random.random() < self.rand_prob:
            np_img = np_img.astype(np.float32) + gauss
            np_img = np.clip(np_img, 0.0, 255.)
            img = Image.fromarray(np_img.astype(np.uint8))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(var_limit={0}, mean={1}, rand_prob={2})'.format(self.var_limit,
                                                                                           self.mean,
                                                                                           self.rand_prob)



