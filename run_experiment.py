"""一键复现60组配对实验和循环平移消融，结果保存在 results/。"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time
import numpy as np
from PIL import Image, __version__ as pillow_version
from denoise import wavelet_denoise, cycle_denoise, mean_filter, psnr, ssim

ROOT = Path(__file__).resolve().parent
METHODS = ['noisy', 'mean3', 'visu', 'bayes', 'cs4', 'cs16', 'cs64']
IMAGES = ['camera', 'chelsea', 'coffee', 'astronaut']
SIGMAS = [10, 20, 30]
SEEDS = [0, 1, 2, 3, 4]


def load_image(name):
    """短边缩放到256后居中裁剪，不拉伸纵横比。灰度转换由Pillow完成。"""
    im = Image.open(ROOT/'data'/(name+'.png')).convert('L')
    w, h = im.size
    ratio = 256/min(w, h)
    im = im.resize((round(w*ratio), round(h*ratio)), Image.Resampling.LANCZOS)
    x, y = (im.width-256)//2, (im.height-256)//2
    return np.asarray(im.crop((x, y, x+256, y+256)), dtype=np.float64)/255


def apply_method(noisy, method):
    if method == 'noisy': return noisy
    if method == 'mean3': return mean_filter(noisy)
    if method in ('visu', 'bayes'): return wavelet_denoise(noisy, method)
    return cycle_denoise(noisy, {'cs4': 2, 'cs16': 4, 'cs64': 8}[method])


def main():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/'examples').mkdir(exist_ok=True)
    records = []
    # 预热不计时；只计去噪函数，不包括指标、读图和写图。
    for method in METHODS:
        apply_method(np.zeros((256, 256)), method)
    for idx, name in enumerate(IMAGES):
        clean = load_image(name)
        np.save(out/'examples'/(name+'_clean.npy'), clean)
        for sigma in SIGMAS:
            for seed in SEEDS:
                actual_seed = 20261101 + idx*10000 + sigma*100 + seed
                noise = np.random.default_rng(actual_seed).normal(0, sigma/255, clean.shape)
                # 加噪后不截断，以保持 AWGN 模型；所有算法完成后统一截断再评价。
                noisy = clean + noise
                for method in METHODS:
                    start = time.perf_counter()
                    estimate = apply_method(noisy, method)
                    elapsed = (time.perf_counter()-start)*1000
                    estimate = np.clip(estimate, 0, 1)
                    records.append(dict(image=name, sigma=sigma, seed=seed,
                        actual_seed=actual_seed, method=method, psnr=psnr(clean, estimate),
                        ssim=ssim(clean, estimate), mse=float(np.mean((clean-estimate)**2)), ms=elapsed))
                    if sigma == 20 and seed == 0:
                        np.save(out/'examples'/(name+'_'+method+'.npy'), estimate)
                print(name, sigma, seed, '完成', flush=True)
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    grouped = []
    for sigma in SIGMAS:
        for method in METHODS:
            rows = [r for r in records if r['sigma'] == sigma and r['method'] == method]
            item = dict(sigma=sigma, method=method, n=len(rows))
            for key in ['psnr', 'ssim', 'mse', 'ms']:
                vals = [r[key] for r in rows]
                item[key] = float(np.mean(vals)); item[key+'_std'] = float(np.std(vals, ddof=1))
            item['ms_median'] = float(np.median([r['ms'] for r in rows]))
            grouped.append(item)
    (out/'summary.json').write_text(json.dumps(grouped, indent=2), encoding='utf-8')
    metadata = dict(python=sys.version, numpy=np.__version__, pillow=pillow_version,
        platform=platform.platform(), processor=platform.processor(), size=[256, 256],
        levels=3, shifts=[1, 4, 16, 64], sigmas=SIGMAS, seeds=SEEDS,
        actual_seed_formula='20261101 + image_index*10000 + sigma*100 + seed',
        noise_clipped=False, output_clipped=True, rows=len(records))
    (out/'environment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('结果:', out, '共', len(records), '行')


if __name__ == '__main__': main()
