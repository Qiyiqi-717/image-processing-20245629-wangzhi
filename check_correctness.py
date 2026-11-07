"""检查可逆性、能量守恒、阈值定义、SSIM自比较和完整相位平均的平移一致性。"""
import numpy as np
from denoise import haar_forward, haar_inverse, soft_threshold, cycle_denoise, ssim

rng = np.random.default_rng(7)
x = rng.normal(size=(32, 48))
a, d = haar_forward(x, 3)
assert np.max(np.abs(haar_inverse(a, d)-x)) < 1e-12
energy = np.sum(a*a)+sum(np.sum(b*b) for level in d for b in level)
assert abs(energy-np.sum(x*x)) < 1e-10
np.testing.assert_allclose(soft_threshold(np.array([-3., -1., 0., 1., 3.]), 1), [-2, 0, 0, 0, 2])
assert abs(ssim(x, x)-1) < 1e-12
y = cycle_denoise(x, grid=8)
shifted = np.roll(x, (1, 3), axis=(0, 1))
np.testing.assert_allclose(cycle_denoise(shifted, grid=8), np.roll(y, (1, 3), axis=(0, 1)), atol=1e-12)
assert np.isfinite(cycle_denoise(np.ones((16, 16)), grid=4)).all()
print('PASS: 重构误差、能量守恒、软阈值、SSIM、平移一致性与常量图像检查')
