import cv2
import random
import numpy as np
import skimage
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(4)


def rotation(img, angle: int, *args):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def flip(img, flip_direction: int, *args):
    """
    Flips the image horizontally or vertically
    
    Params
    -----

    img: numpy array of the image from cv2.imread()

    flip_direction = 0 means flip image vertically,
    flip_direction = 1 means flip image horizontally
    """
    return cv2.flip(img, flip_direction)


def image_reflection(img, ratio):
    if not -0.9 <= ratio <= 0.9:
        raise ValueError("Ratio must be between 0.9 and -0.9")
    h, w = img.shape[:2]
    to_shift = int(w*abs(ratio))
    if ratio > 0:
        img = img[:, :w-to_shift, :]
    if ratio < 0:
        img = img[:, -1*to_shift:, :]
    return cv2.copyMakeBorder(img, 0, 0, to_shift, 0, cv2.BORDER_REFLECT)




def zoom(img, value):
    """
    Zooms in to random part from the image

    Params
    ------

    img: numpy array of the image from cv2.imread()

    value: indicates the percentage of the images that we want to zoom to.
    Example: A value of 0.6 will mean to take 60% of the whole image and then we will resize it back to the original size
    
    """
    def fill(img, w, h):
        img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
        return img
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, w, h)
    return img


def blur(img, scale, *args):
    return cv2.blur(img, (scale, scale))


def add_noise_using_imgaug(img: np.ndarray, scale: tuple):
    return iaa.AdditiveGaussianNoise(scale=scale)(image=img)

def rotate_using_imgaug(img: np.ndarray, rotation: int):
    rotate = iaa.Affine(rotate=(-rotation, rotation))
    image_aug = rotate(image=img)
    return image_aug

def crop_using_imgaug(img: np.ndarray, scale: float):
    return iaa.Crop(percent=(0, scale))(image=img)

def crop_and_rotate(img, crop_rate, rotation):
    return rotate_using_imgaug(crop_using_imgaug(img, crop_rate), rotation)

# if __name__ == "__main__":
#     img = cv2.imread(r"E:\python projects\AI\selected 2\magdyy dataset\Train\pineapple\Image_53.jpg", cv2.IMREAD_UNCHANGED)
   
#     # img = crop_using_imgaug(img, 0.25)
#     print(img)
#     cv2.imshow("ddd", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

