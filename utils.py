import torch
import cv2
from scipy import signal
import numpy as np
import random
from collections import OrderedDict, Counter

def _rgb2yuv2(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[16],#16
                  [128],#128
                  [128]])#128
    T = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    yuv[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    yuv[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    yuv[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    yuv = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return yuv

def _yuv2rgb2(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[0],#16
                  [128],#128
                  [128]])#128
    T = np.array([[1, 0, 1.4022],
                  [1, -0.3456, -0.7145],
                  [1, 1.7710, 0]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    R =uint8( 1 * Y + 0  * U + 1.4022* V);
#    G = uint8(1 * Y  -0.3456  * U  -0.7145* V);
#    B= uint8(1 * Y + 1.7710  * U + 0 * V );
#    yuv[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    yuv[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    yuv[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t[:, 0] -= O[0]
    t[:, 1] -= O[1]
    t[:, 2] -= O[2]
    t = np.dot(t, np.transpose(T))
    rgb = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    rgb = np.round(rgb)
    rgb = np.clip(rgb,0,maxVal)
    rgb = rgb.astype(np.uint8)
    return rgb

def _yuv2rgb(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[0],#0
                  [128],#128
                  [128]])#128
    T = np.array([[1, 0, 1.4022],
                  [1, -0.3456, -0.7145],
                  [1, 1.7710, 0]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    R =uint8( 1 * Y + 0  * U + 1.4022* V);
#    G = uint8(1 * Y  -0.3456  * U  -0.7145* V);
#    B= uint8(1 * Y + 1.7710  * U + 0 * V );
#    yuv[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    yuv[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    yuv[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t[:, 0] -= O[0]
    t[:, 1] -= O[1]
    t[:, 2] -= O[2]
    t = np.dot(t, np.transpose(T))
    yuv = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 255:
        yuv = np.round(yuv)
        yuv = np.clip(yuv, 0, 255).astype(np.uint8)
    else:
        yuv = np.clip(yuv, 0, 1) # 保持为 float32
    return yuv

def _rgb2yuv(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[0],#0
                  [128],#128
                  [128]])#128
    T = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    yuv[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    yuv[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    yuv[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    yuv = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return yuv

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bPSNR(img1, img2, crop_border):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def remove_module(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def bit_plane_slicing(x, bit_mask='11110000'):
    m = int(bit_mask, 2)
    masks = [m, 255-m]
    images = []
    for mask in masks:
        images.append((x.type(torch.LongTensor) & mask).type(torch.FloatTensor).to(x.device))
    return images

    
def decode_bit_mask(bit_mask):

    bit_counts = Counter(bit_mask) 
    msb_step= 2**(bit_mask[::-1].find("1"))
    return bit_counts["1"], bit_counts["0"], msb_step, 1

def floor_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    # # new: Remove 128
    # input[input > 127.5] = 127.5
    forward_value = torch.floor(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def _rgb2ycbcr(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    ycbcr[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    ycbcr[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    ycbcr[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return ycbcr


def _ycbcr2rgb(ycbcr, maxVal=255):
    """
    YCbCr to RGB

    param:
        ycbcr: numpy array，YCbCr format (H,W,3)
        maxVal: maximal value，1 represent [0,1]， 255 represent [0,255]

    return:
        RGB format (H,W,3)
    """
    # Inverse transformation matrix and offset
    O = np.array([[16],
                  [128],
                  [128]])

    T_inv = np.array([[1.164383561643836, 0.000000000000000, 1.596026785714286],
                      [1.164383561643836, -0.391762290094914, -0.812967647237771],
                      [1.164383561643836, 2.017232142857143, 0.000000000000000]])

    if maxVal == 1:
        O = O / 255.0

    # subtract offset
    t = np.reshape(ycbcr, (ycbcr.shape[0] * ycbcr.shape[1], ycbcr.shape[2]))
    t[:, 0] -= O[0]
    t[:, 1] -= O[1]
    t[:, 2] -= O[2]

    # Applying the inverse transformation matrix
    t = np.dot(t, np.transpose(T_inv))

    # Reshape to the original shape and trim to the effective range
    rgb = np.reshape(t, [ycbcr.shape[0], ycbcr.shape[1], ycbcr.shape[2]])
    rgb = np.round(rgb)
    rgb = np.clip(rgb, 0, maxVal)

    return rgb

def _load_img_array(path, color_mode='RGB', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    ## To 0-1
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x

def modcrop(image, modulo):
    if len(image.shape) == 2:
        sz = image.shape
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1]]
    elif image.shape[2] == 3:
        sz = image.shape[0:2]
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1], :]
    else:
        raise NotImplementedError
    return image



    
def PSNR(y_true, y_pred, shave_border=4):
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255. / rmse)

def calculate_rgb_psnr(img1, img2, shave_border=4):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = PSNR(img1[:, :, i], img2[:, :, i], shave_border)
        sum_psnr += this_psnr
    return sum_psnr / n_channels

def cPSNR(y_true, y_pred, shave_border=0):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border, :]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255. / rmse)

def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim

def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def add_uniform_noise(image: np.ndarray, prob=0.05, vaule=255):
    """
    A random mask between 0 and 1 is generated and used as salt-and-pepper noise.
    :param image: image
    :param prob: Noise ratio
    :param vaule: noise value
    :return:
    """
    h, w = image.shape[:2]
    noise = np.random.uniform(low=0.0, high=1.0, size=(h, w)).astype(dtype=np.float32)  # 产生高斯噪声
    mask = np.zeros(shape=(h, w), dtype=np.uint8) + vaule
    index = noise > prob
    mask = mask * (~index)
    output = image * index[:, :, np.newaxis] + mask[:, :, np.newaxis]
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output