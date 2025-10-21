import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_two_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return imgs[:2] if len(imgs) >= 2 else []

def orb_align(ref, mov, min_matches=10):
    gray1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d2, d1, k=2)  # match mov -> ref
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None

    pts_mov = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_mov, pts_ref, cv2.RANSAC, 5.0)
    return H

def warp_to_ref(ref, mov, H):
    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(mov, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped

def center_pad_or_crop(a, target_shape):
    th, tw = target_shape[:2]
    h, w = a.shape[:2]
    # if bigger -> crop center, if smaller -> pad equally
    y0 = max(0, (h - th)//2)
    x0 = max(0, (w - tw)//2)
    cropped = a[y0:y0+min(th,h), x0:x0+min(tw,w)]
    pad_top = max(0, (th - cropped.shape[0])//2)
    pad_bottom = th - cropped.shape[0] - pad_top
    pad_left = max(0, (tw - cropped.shape[1])//2)
    pad_right = tw - cropped.shape[1] - pad_left
    return cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

# helper to ensure we never overwrite originals — return a unique filepath
def _unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def align_pair(path1, path2, out_suffix='_aligned_copy'):
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return False, 'read_failed'

    H = orb_align(img1, img2, min_matches=12)
    # construct output paths in same directory as inputs, avoid overwriting
    dir1 = os.path.dirname(path1)
    dir2 = os.path.dirname(path2)
    name1, ext1 = os.path.splitext(os.path.basename(path1))
    name2, ext2 = os.path.splitext(os.path.basename(path2))

    if H is not None:
        warped2 = warp_to_ref(img1, img2, H)
        # keep both images same size as ref
        out1 = center_pad_or_crop(img1, img1.shape)
        out2 = center_pad_or_crop(warped2, img1.shape)
        base1 = os.path.join(dir1, f"{name1}{out_suffix}{ext1}")
        base2 = os.path.join(dir2, f"{name2}{out_suffix}{ext2}")
        base1 = _unique_path(base1)
        base2 = _unique_path(base2)
        cv2.imwrite(base1, out1)
        cv2.imwrite(base2, out2)
        return True, 'homography'
    else:
    # fallback: center-resize/pad to same size (use larger dims)
        th = max(img1.shape[0], img2.shape[0])
        tw = max(img1.shape[1], img2.shape[1])
        o1 = center_pad_or_crop(img1, (th, tw, 3))
        o2 = center_pad_or_crop(img2, (th, tw, 3))
        base1 = os.path.join(dir1, f"{name1}{out_suffix}{ext1}")
        base2 = os.path.join(dir2, f"{name2}{out_suffix}{ext2}")
        base1 = _unique_path(base1)
        base2 = _unique_path(base2)
        cv2.imwrite(base1, o1)
        cv2.imwrite(base2, o2)
        return True, 'fallback_pad'

def main(root, recursive=False):
    folders = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            folders.append(dirpath)
    else:
        folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for folder in tqdm(sorted(folders)):
        imgs = find_two_images(folder)
        if len(imgs) < 2:
            continue
        ok, reason = align_pair(imgs[0], imgs[1])
        # short log to stdout
        print(f"{folder}: {os.path.basename(imgs[0])}, {os.path.basename(imgs[1])} -> {reason}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align image pairs in folders using ORB+homography.')
    parser.add_argument('root', nargs='?', default='data/youtube/tmp', help='root folder containing pair folders')
    parser.add_argument('--recursive', action='store_true', help='walk subfolders recursively')
    args = parser.parse_args()
    main(args.root, args.recursive)
