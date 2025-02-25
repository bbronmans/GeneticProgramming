""""
Preprocess_img class. All image preprocessing methods are located here.
"""
import numpy as np


def random_hflip_img(img):
    """Flips the input x horizontally with 50% probability."""
    if np.random.rand() > 0.5:
        return np.fliplr(img)
    return img


def zero_pad_and_crop_img(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.
    Args:
      img: numpy image that will be zero padded and cropped.
      amount: amount of zeros to pad `img` with horizontally and verically.
    Returns:
      The cropped zero padded img. The returned numpy array will be of the same
      shape as `img`.
    """
    padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                           img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount: img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
    return new_img


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
    Args:
      img_height: Height of image cutout mask will be applied to.
      img_width: Width of image cutout mask will be applied to.
      num_channels: Number of channels in the image.
      size: Size of the zeros mask.
    Returns:
      A mask of shape `img_height` x `img_width` with all ones except for a
      square of zeros of shape `size` x `size`. This mask is meant to be
      elementwise multiplied with the original image. Additionally returns
      the `upper_coord` and `lower_coord` which specify where the cutout mask
      will be applied.
    """
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
        zeros)
    return mask, upper_coord, lower_coord


def cutout_img(img, size=2):
    """Apply cutout with mask of shape `size` x `size` to `img`.
    The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
    This operation applies a `size`x`size` mask of zeros to a random location
    within `img`.
    Args:
      img: Numpy image that cutout will be applied to.
      size: Height/width of the cutout mask that will be
    Returns:
      A numpy tensor that is the result of applying the cutout mask to `img`.
    """
    img_height, img_width, num_channels = (img.shape[0], img.shape[1],
                                           img.shape[2])
    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask


def augment_img(img):
    """Augments 'img' by applying random horizontal flips, cutout and random crops."""
    img = random_hflip_img(img)
    img = cutout_img(img, size=12)
    img = zero_pad_and_crop_img(img)
    return img
