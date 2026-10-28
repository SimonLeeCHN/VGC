import cv2
import numpy as np
from typing import Optional, Tuple
import os

# Global config: respect fixed thresholds if enabled
USE_FIXED_DIFF = True
FIXED_LO_DIFF = 10
FIXED_UP_DIFF = 40

def compute_thresholds_from_seed(optMat: np.ndarray, seed_val: int) -> Tuple[int, int, int]:
    """
    Compute floodFill thresholds based on a seed pixel value.
    Input:
      - optMat: grayscale image (uint8, single channel)
      - seed_val: pixel value at the seed point
    Returns:
      - (thred, loDiff, upDiff) for floodFill
    """
    # simple implementation: use fixed diff around the seed value
    thred = int(seed_val)
    loDiff = FIXED_LO_DIFF
    upDiff = FIXED_UP_DIFF
    return thred, loDiff, upDiff


def _AIM_FloodFill(optMat: np.ndarray, erode_size: int = 8, show_process_view: bool = False) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int]], Optional[Tuple[int,int,int]]]:
    """
    Python translation of the provided C++ _AIM_FloodFill routine.
    Input:
      - optMat: grayscale image (uint8, single channel)
      - erode_size: kernel size for erosion
      - show_process_view: if True, will call cv2.imshow for intermediate steps
    Returns:
      - flood_mask: uint8 mask (0/255) same size as optMat, or None on failure
      - bbox: (x, y, w, h) of the flood-filled rect, or None
      - seed: (x, y) seed point used for floodFill, or None
      - threds: (thred, loDiff, upDiff) used for floodFill, or None
    """
    if optMat is None:
        return None, None, None, None
    if len(optMat.shape) != 2:
        # require single channel
        raise ValueError("_AIM_FloodFill optMat must be single-channel grayscale (CV_8UC1)")

    src = optMat.copy()
    # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        src = clahe.apply(src)
    except Exception:
        # if CLAHE not supported or fails, continue with original source
        pass
    # median blur (use the returned image in Python API)
    try:
        src = cv2.medianBlur(src, 3)
    except Exception:
        pass
    # Otsu inverted threshold: indentify dark regions as foreground
    _, otsuInv = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show_process_view:
        cv2.imshow('1-二值化图像', otsuInv)

    # Erode to close gaps
    erode_k = max(1, int(erode_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_k, erode_k))
    otsuInv = cv2.erode(otsuInv, kernel)
    if show_process_view:
        cv2.imshow('2-缝隙填补', otsuInv)

    # find external contours
    contours, _ = cv2.findContours(otsuInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # no contours -> fail
        return None, None, None, None

    # find largest contour by area
    max_idx = int(np.argmax([abs(cv2.contourArea(c)) for c in contours]))
    max_contour = contours[max_idx]

    # draw filled mask of largest contour
    matMaxMask = np.zeros_like(otsuInv)
    cv2.drawContours(matMaxMask, contours, max_idx, 255, thickness=-1)

    if show_process_view:
        show_img = cv2.cvtColor(optMat, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(show_img, contours, max_idx, (255,0,0), thickness=1)
        cv2.imshow('3-最大连通域', show_img)

    # find darkest and brightest points inside the max contour
    ys, xs = np.where(matMaxMask == 255)
    if ys.size == 0:
        return None, None, None, None
    vals = optMat[ys, xs]
    min_idx = int(np.argmin(vals))
    max_idx_loc = int(np.argmax(vals))
    min_y, min_x = int(ys[min_idx]), int(xs[min_idx])
    max_y, max_x = int(ys[max_idx_loc]), int(xs[max_idx_loc])
    min_val = int(vals[min_idx])
    max_val = int(vals[max_idx_loc])

    if show_process_view:
        viz = cv2.cvtColor(optMat, cv2.COLOR_GRAY2BGR)
        cv2.circle(viz, (min_x, min_y), 5, (255,0,0), -1)
        cv2.circle(viz, (max_x, max_y), 5, (0,0,255), -1)
        cv2.imshow('4-连通域中最暗点(蓝)与最亮点(红)', viz)

    # prepare floodFill mask (with 2-pixel border)
    h, w = optMat.shape[:2]
    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)

    flood_src = src.copy()
    flood_rect = None
    # flags: 8-connectivity | (maskFillValue<<8) | FIXED_RANGE | MASK_ONLY
    flags = 8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    # compute thresholds based on darkest point
    thred, loDiff, upDiff = compute_thresholds_from_seed(optMat, min_val)
    # respect global fixed config
    if USE_FIXED_DIFF:
        loDiff = FIXED_LO_DIFF
        upDiff = FIXED_UP_DIFF

    # perform floodFill (seed point is min location)
    seed_pt = (int(min_x), int(min_y))
    try:
        retval, img_after, ffmask, rect = cv2.floodFill(flood_src, flood_mask, seed_pt, 255, loDiff, upDiff, flags)
        flood_mask_result = ffmask[1:-1, 1:-1]
        # normalize to 0/255
        result_mask = (flood_mask_result > 0).astype(np.uint8) * 255
        flood_rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    except Exception:
        return None, None, None, None

    if show_process_view:
        cv2.imshow('5-以最暗点递归搜索到的区域', flood_mask)

    return result_mask, flood_rect, (min_x, min_y), (thred, loDiff, upDiff)


# integrate with existing process_image: call _AIM_FloodFill when no MICF available
def process_image(image_path: str, erode_size: int = 8, clahe_tileGridSize=(2, 2), clahe_clipLimit: float = 2.0) -> Tuple[Optional[object], Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int, int, int]]]:
    """
    Process a single image path:
    - read image and convert to gray
    - call _AIM_FloodFill (translated from Qt code) to get flood mask and rect
    - return flood mask, seed and thresholds
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return None, None, None, None
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Optionally use MICF if available, but follow Qt code by default
    try:
        flood_mask, flood_rect, seed, threds = _AIM_FloodFill(gray, erode_size=erode_size, show_process_view=False)
    except Exception:
        flood_mask, flood_rect, seed, threds = None, None, None, None

    return None, flood_mask, seed, threds


if __name__ == '__main__':
    print('请选择模式：\n1 - 处理单张图片\n2 - 处理文件夹内所有图片')
    mode = input('输入模式(1/2): ').strip()
    if mode == '1':
        img_path = input('请输入单张图片路径: ').strip()
        if not os.path.isfile(img_path):
            print('路径无效或不是文件:', img_path)
        else:
            _, mask, seed, threds = process_image(img_path)
            if mask is None:
                print('处理失败或未得到 mask。')
            else:
                dirn = os.path.dirname(img_path) or '.'
                base = os.path.splitext(os.path.basename(img_path))[0]
                outname = os.path.join(dirn, base + '_FF.png')
                cv2.imwrite(outname, mask)
                print('已保存结果到:', outname)
    elif mode == '2':
        folder = input('请输入要处理的文件夹路径: ').strip()
        if not os.path.isdir(folder):
            print('无效文件夹路径:', folder)
        else:
            outdir = os.path.join(folder, 'FF_binary')
            os.makedirs(outdir, exist_ok=True)
            img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in img_exts:
                    continue
                fpath = os.path.join(folder, f)
                _, mask, seed, threds = process_image(fpath)
                if mask is None:
                    print(f'{f}: 未生成 mask，跳过')
                    continue
                outname = os.path.join(outdir, os.path.splitext(f)[0] + '_FF.png')
                cv2.imwrite(outname, mask)
                print('已保存:', outname)
            print('批量处理完成，结果保存在:', outdir)
    else:
        print('无效输入，程序退出。')
