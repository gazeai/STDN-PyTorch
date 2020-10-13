from __future__ import division
from PIL import Image, ImageOps, ImageEnhance

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np

from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        if type(imgs) != list:
            img = imgs
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                imgs = t(imgs)
            return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        out = list(F.to_tensor(pic) for pic in pics)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
            ``int``, ``float``, ``short``).
    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, *pics):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL Image: Image converted to PIL Image.
        """

        # print(type(pics[0][1]), type(pics[0][0]))
        for p in range(len(pics[0])):
            # print("image shape", pics[0][p].shape)
            if len(pics[0][p].shape) > 2:
                pics[0][p] = F.to_pil_image(pics[0][p], self.mode)
        # out = list(F.to_pil_image(pic, self.mode) for pic in pics[0])
        out = pics[0]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class ScaleTensor(object):
    def __init__(self, base_key, trans_key, size=256):
        self.base_key = base_key
        self.trans_key = trans_key
        self.size = size
    def __call__(self, imgs):

        base = imgs[self.base_key]
        to_scale = imgs[self.trans_key]
        # print("Before scaling")
        # print(imgs[self.trans_key])
        # print("After Scaling")
        h, w, _ = base.shape if type(base) == np.ndarray else base.size
        # print(h, w, _)
        to_scale[:, 0] = to_scale[:, 0]*self.size/w
        to_scale[:, 1] = to_scale[:, 1]*self.size/h
        imgs[self.trans_key] = to_scale
        # print(imgs[self.base_key])
        # print(imgs[self.trans_key])
        return imgs


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        print("Random horizontal flip init")
        self.p = p
        self.lm_reverse_list = np.array([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                    27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                                    28, 29, 30, 31, 36, 35, 34, 33, 32,
                                    46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41,
                                    55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66],
                                   np.int32) - 1

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        # print("ekhane ki asheeeeeeeeeeee?")
        if np.random.rand() > self.p:
            # print(img.shape for img in imgs)
            # for img in imgs:
            #     print(img.shape, type(img))
            for i in range(len(imgs)):
                if i % 2 == 0:
                    # print(imgs[i].shape, type(imgs[i]))
                    # imgs[i] = Image.fromarray(imgs[i])
                    # for img in imgs:
                        # print(img.size, type(img))
                    imgs[i] = F.hflip(imgs[i])
                    # print(np.array(imgs[i]), np.array(imgs[i]).shape)
                elif i % 2 == 1:
                    imgs[i][:, 0] = 256 - imgs[i][:, 0]
                    imgs[i] = imgs[i][self.lm_reverse_list, :]
            out = imgs
        else:
            out = imgs
        # print("flip hoy??? :O")
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
