from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MICFResult:
	"""最大连通域外接矩形结果。"""
	x: int
	y: int
	width: int
	height: int


def aim_max_internal_connect_filed(
	opt_mat: np.ndarray,
	erode_size: int = 8,
    show_process_view: bool = False,
    window_prefix: str = "",
    clahe_clipLimit: float = 2.0,
    clahe_tileGridSize: tuple[int, int] = (2, 2),
) -> Optional[tuple[MICFResult, np.ndarray]]:
	"""
	自动压痕测量 - 寻找最大内部连通域方法（Python 版）。

	参数:
		opt_mat: 输入图像，要求为单通道 8bit（uint8）。
		erode_size: 腐蚀核尺寸，对应 C++ 中 imaErodeSize。
		show_process_view: 是否显示中间过程图。
		window_prefix: 过程窗口名前缀。

	返回:
		MICFResult: 最大连通域外接矩形。
		None: 输入非法或未找到轮廓。
	"""
	if opt_mat is None:
		print("aim_max_internal_connect_filed: opt_mat is None")
		return None

	if opt_mat.ndim != 2:
		print("aim_max_internal_connect_filed: opt_mat must be CV_8UC1 (single channel)")
		return None

	if opt_mat.dtype != np.uint8:
		print("aim_max_internal_connect_filed: opt_mat dtype must be uint8")
		return None

	# 先做 CLAHE 提升局部对比度，再做中值滤波
	# CLAHE 可增强压痕暗部的对比，使后续二值化效果更好
	# 对传入的参数做简单校验，保证类型与范围合理
	try:
		_clip = float(clahe_clipLimit)
		if _clip <= 0:
			print(f"clahe_clipLimit must be > 0, got {clahe_clipLimit}. Using default 2.0")
			_clip = 2.0
	except Exception:
		print(f"Invalid clahe_clipLimit: {clahe_clipLimit}. Using default 2.0")
		_clip = 2.0
	try:
		_tile = (int(clahe_tileGridSize[0]), int(clahe_tileGridSize[1]))
		if _tile[0] <= 0 or _tile[1] <= 0:
			print(f"clahe_tileGridSize elements must be > 0, got {clahe_tileGridSize}. Using default (8,8)")
			_tile = (8, 8)
	except Exception:
		print(f"Invalid clahe_tileGridSize: {clahe_tileGridSize}. Using default (8,8)")
		_tile = (8, 8)
	clahe = cv2.createCLAHE(clipLimit=_clip, tileGridSize=_tile)
	work_clahe = clahe.apply(opt_mat)
	if show_process_view:
		win_name = f"{window_prefix}0-CLAHE结果"
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.imshow(win_name, work_clahe)
		cv2.waitKey(1)

	# # 灰度调整：保留50以下不变，50-255按对数曲线提升亮度
	# adjusted = work_clahe.astype(np.float32)
	# low_mask = adjusted < 50
	# adjusted[~low_mask] = 50 + (255 - 50) * (np.log(1 + (adjusted[~low_mask] - 50)) / np.log(1 + (255 - 50)))
	# adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
	# work_clahe = adjusted

	# 中值滤波（与 C++ 保持一致）
	work = cv2.medianBlur(work_clahe, 3)

	# Otsu 反阈值：压痕暗部 -> 亮部
	_, otsu_inv_mat = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	if show_process_view:
		win_name = f"{window_prefix}1-二值化图像"
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.imshow(win_name, otsu_inv_mat)
		cv2.waitKey(1)

	# 腐蚀，封闭压痕边沿，避免边缘划痕连通
	erode_size = max(1, int(erode_size))
	erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
	otsu_inv_mat = cv2.erode(otsu_inv_mat, erode_kernel)
	if show_process_view:
		win_name = f"{window_prefix}2-缝隙填补"
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.imshow(win_name, otsu_inv_mat)
		cv2.waitKey(1)

	# 找外侧轮廓
	contours_outside, _ = cv2.findContours(
		otsu_inv_mat,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_NONE,
	)
	if len(contours_outside) == 0:
		print("aim_max_internal_connect_filed: len(contours_outside) == 0")
		return None

	# 查找最大轮廓，并获得其索引（避免直接用 list.index 比较 numpy 数组导致的歧义）
	max_index = max(range(len(contours_outside)), key=lambda i: abs(cv2.contourArea(contours_outside[i])))
	max_contour = contours_outside[max_index]

	if show_process_view:
		mat_max_filed_show = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
		# 使用已经计算出的索引绘制对应轮廓
		cv2.drawContours(mat_max_filed_show, contours_outside, max_index, (255, 0, 0), -1)
		win_name = f"{window_prefix}3-最大连通域"
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.imshow(win_name, mat_max_filed_show)
		cv2.waitKey(1)

	# 生成mask，轮廓内白色，外部黑色
	mask = np.zeros_like(opt_mat, dtype=np.uint8)
	cv2.drawContours(mask, [max_contour], -1, 255, thickness=-1)

	# 求最大连通域的外接矩形
	x, y, w, h = cv2.boundingRect(max_contour)
	return MICFResult(x=x, y=y, width=w, height=h), mask


if __name__ == "__main__":
	def process_images_in_folder(folder_path, erode_size=8, clahe_tileGridSize=(2,2), clahe_clipLimit=2, show_process_view=False):
		"""
		批量处理文件夹下所有图像文件，自动转灰度并执行MICF操作。
		:param folder_path: 文件夹路径
		:param erode_size: 腐蚀核尺寸
		:param show_process_view: 是否显示中间过程
		:return: 字典，键为文件名，值为(MICFResult或None, mask或None)
		"""
		import os
		results = {}
		img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
		for fname in os.listdir(folder_path):
			fpath = os.path.join(folder_path, fname)
			if not os.path.isfile(fpath):
				continue
			ext = os.path.splitext(fname)[1].lower()
			if ext not in img_exts:
				continue
			img = cv2.imread(fpath)
			if img is None:
				print(f"无法读取图像: {fname}")
				results[fname] = (None, None)
				continue
			# 自动转灰度
			if len(img.shape) == 3 and img.shape[2] == 3:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			else:
				img_gray = img
			result, mask = aim_max_internal_connect_filed(
				img_gray,
				clahe_tileGridSize=clahe_tileGridSize,
				clahe_clipLimit=clahe_clipLimit,
				erode_size=erode_size,
				show_process_view=show_process_view,
				window_prefix=fname+"-",
			)
			results[fname] = (result, mask)
		return results

	# 选择运行模式：单图像自检 or 文件夹批量处理
	mode = input("输入1进行单图像自检，输入2批量处理文件夹: ")
	if mode.strip() == '1':
		img_path = input("请输入单张图片路径: ")
		img = cv2.imread(img_path)
		if img is None:
			print(f"无法读取图像: {img_path}")
		else:
			# 自动转灰度
			if len(img.shape) == 3 and img.shape[2] == 3:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			else:
				img_gray = img
			# 默认使用函数的 CLAHE 参数，如果需要可修改后重启脚本
			result, mask = aim_max_internal_connect_filed(
				img_gray,
				clahe_tileGridSize=(2,2),
				clahe_clipLimit=2,
				erode_size=8,
				show_process_view=True,
				window_prefix="single-",
			)
			print(result)
			# 可选：保存mask
			if mask is not None:
				cv2.imwrite("single_MICF_mask.jpg", mask)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	elif mode.strip() == '2':
		folder = input("请输入要处理的文件夹路径: ")
		res = process_images_in_folder(folder, show_process_view=False)
		# 在当前代码根目录下创建一个结果文件夹来存储 mask 和汇总
		import os, csv
		code_root = os.path.dirname(__file__)
		out_dir = os.path.join(code_root, "MICF_results")
		os.makedirs(out_dir, exist_ok=True)
		summary_path = os.path.join(out_dir, "results_summary.csv")
		with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["filename", "x", "y", "width", "height"])
			for fname, (r, mask) in res.items():
				print(f"{fname}: {r}")
				if r is not None:
					writer.writerow([fname, r.x, r.y, r.width, r.height])
				else:
					writer.writerow([fname, "", "", "", ""])
				# 保存 mask 到 out_dir
				if mask is not None:
					mask_filename = f"{os.path.splitext(fname)[0]}_MICF_mask.png"
					mask_path = os.path.join(out_dir, mask_filename)
					cv2.imwrite(mask_path, mask)
		print(f"Results saved to folder: {out_dir}")
	else:
		print("无效输入")
