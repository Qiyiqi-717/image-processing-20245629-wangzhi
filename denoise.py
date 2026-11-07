"""仅依赖 NumPy 的 Haar 小波去噪；不使用干净参考图像。"""
from __future__ import annotations
import numpy as np


def haar_forward(image, levels=3):
    """正交归一化二维 Haar。details 按从细到粗存放三个方向的细节。"""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or levels < 1 or any(n % (2**levels) for n in a.shape):
        raise ValueError('二维图像的尺寸须为 2**levels 的倍数，levels >= 1')
    details = []
    for _ in range(levels):
        # 先沿列分解，再沿行分解；每一步除以 sqrt(2) 保持能量。
        low = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
        high = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
        ll = (low[0::2] + low[1::2]) / np.sqrt(2)
        dh = (low[0::2] - low[1::2]) / np.sqrt(2)
        dv = (high[0::2] + high[1::2]) / np.sqrt(2)
        dd = (high[0::2] - high[1::2]) / np.sqrt(2)
        details.append((dh, dv, dd))
        a = ll
    return a, details


def haar_inverse(approx, details):
    """不做阈值操作时，能够在浮点精度内精确恢复原图。"""
    a = approx.copy()
    for dh, dv, dd in reversed(details):
        low = np.empty((a.shape[0]*2, a.shape[1]))
        high = np.empty_like(low)
        low[0::2], low[1::2] = (a+dh)/np.sqrt(2), (a-dh)/np.sqrt(2)
        high[0::2], high[1::2] = (dv+dd)/np.sqrt(2), (dv-dd)/np.sqrt(2)
        a = np.empty((low.shape[0], low.shape[1]*2))
        a[:, 0::2], a[:, 1::2] = (low+high)/np.sqrt(2), (low-high)/np.sqrt(2)
    return a


def soft_threshold(coeff, threshold):
    return np.sign(coeff) * np.maximum(np.abs(coeff)-threshold, 0)


def wavelet_denoise(image, method='bayes', levels=3):
    """VisuShrink 或 BayesShrink；由最细对角子带估计噪声标准差。"""
    a, details = haar_forward(image, levels)
    sigma = float(np.median(np.abs(details[0][2])) / 0.6744897501960817)
    var_noise = sigma*sigma
    universal = sigma*np.sqrt(2*np.log(image.size))
    filtered = []
    for bands in details:
        output = []
        for band in bands:
            if method == 'visu':
                threshold = universal
            elif method == 'bayes':
                # 正交变换下，观测子带二阶矩近似为信号方差加噪声方差。
                var_signal = max(float(np.mean(band*band))-var_noise, 0.0)
                threshold = var_noise/np.sqrt(var_signal) if var_signal > 1e-15 else np.inf
            else:
                raise ValueError('method 必须为 visu 或 bayes')
            output.append(soft_threshold(band, threshold))
        filtered.append(tuple(output))
    return haar_inverse(a, filtered)


def cycle_denoise(image, grid=4, levels=3):
    """平移 -> BayesShrink -> 反平移 -> 平均。grid=4 表示 16 次。"""
    if grid < 1:
        raise ValueError('grid 必须为正整数')
    result = np.zeros_like(image, dtype=np.float64)
    for dy in range(grid):
        for dx in range(grid):
            shifted = np.roll(image, (dy, dx), axis=(0, 1))
            recovered = wavelet_denoise(shifted, 'bayes', levels)
            result += np.roll(recovered, (-dy, -dx), axis=(0, 1))
    return result/(grid*grid)


def mean_filter(image):
    """3x3 均值滤波，使用循环边界。"""
    return sum(np.roll(image, (dy, dx), axis=(0, 1))
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))/9


def psnr(reference, image):
    mse = float(np.mean((reference-image)**2))
    return float(10*np.log10(1/mse)) if mse else float('inf')


def _gaussian_valid(image):
    """11x11、sigma=1.5 的归一化高斯窗口，只保留有效区域。"""
    weights = np.exp(-(np.arange(-5, 6)**2)/(2*1.5**2))
    weights /= weights.sum()
    h, w = image.shape
    temp = sum(weights[i]*image[i:i+h-10, :] for i in range(11))
    return sum(weights[i]*temp[:, i:i+w-10] for i in range(11))


def ssim(reference, image):
    """Wang 等人的单尺度 SSIM；总体加权矩，data_range=1，去掉5像素边界。"""
    ux, uy = _gaussian_valid(reference), _gaussian_valid(image)
    vx = np.maximum(_gaussian_valid(reference**2)-ux**2, 0)
    vy = np.maximum(_gaussian_valid(image**2)-uy**2, 0)
    cov = _gaussian_valid(reference*image)-ux*uy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2*ux*uy+c1)*(2*cov+c2))/((ux**2+uy**2+c1)*(vx+vy+c2))
    return float(score.mean())


if __name__ == '__main__':
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description='使用16次循环平移的小波去噪处理自己的图片')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='denoised.png')
    parser.add_argument('--grid', type=int, default=4)
    args = parser.parse_args()
    image = np.array(Image.open(args.input).convert('L'), dtype=float)/255
    h, w = image.shape
    padded = np.pad(image, ((0, (-h)%8), (0, (-w)%8)), mode='symmetric')
    out = np.clip(cycle_denoise(padded, grid=args.grid)[:h, :w], 0, 1)
    Image.fromarray(np.uint8(np.rint(out*255))).save(args.output)
    print('已保存:', args.output)
