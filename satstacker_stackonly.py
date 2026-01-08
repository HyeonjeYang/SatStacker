# Copyright (c) 2026 Penta0308
import glob
import os
from datetime import datetime

import cv2
import numpy as np

def lab_median_stacking(frames_dir: str, sat_sigma: float = -3.0) -> np.ndarray:
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.tif")))
    print("# 4   Stacking.")

    first_img = cv2.imread(frame_files[0], cv2.IMREAD_UNCHANGED)
    h, w, _ = first_img.shape

    stack = np.zeros((len(frame_files), h, w, 3), dtype=np.float32)

    for i, f in enumerate(frame_files):
        print(f"  4 1 Reading {f}")
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        img_float = img.astype(np.float32) / 65535.0
        stack[i] = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)

    print("  4 2 Extracting Luminosity.")
    l_channel = stack[:, :, :, 0]
    l_avg = np.mean(l_channel, axis=0)
    l_std = np.std(l_channel, axis=0) + 1e-6

    print("  4 3 Calculating Weight.")
    z_score_raw = (l_channel - l_avg) / l_std
    weights = 1.0 / (1.0 + np.exp(-5.0 * np.sign(sat_sigma) * (z_score_raw - sat_sigma)))

    print("  4 4 Rejecting L > 99.99percent.")
    weights[l_channel > 99.99] = 0

    print("  4 5 Creating Mask.")
    max_idx = np.argmax(weights, axis=0)
    alpha = np.max(weights, axis=0)

    print("  4 6 Applying Mask.")
    yy, xx = np.indices((h, w))
    sat_candidate = stack[max_idx, yy, xx]
    median_bg = np.median(stack, axis=0)
    final_lab = (alpha[..., None] * sat_candidate) + ((1 - alpha[..., None]) * median_bg)

    print("  4 8 Converting to RGB.")
    result_bgr_float = cv2.cvtColor(final_lab, cv2.COLOR_Lab2BGR)
    result_16bit = np.clip(result_bgr_float * 65535.0, 0, 65535).astype(np.uint16)

    print("# 4   Succeeded.")
    return result_16bit

if __name__ == "__main__":
    frames_dir = input("?     Drag previous frame folder (inside job) here [path]: ").strip().replace("'", "").replace('"', "")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    sat_sigma = input("?     Threshold for Satellite detection. Negative for darker than background(moon, etc.). Default -3.0 [float] SD: ")
    if sat_sigma == "":
        sat_sigma = -3.0

    result_image = lab_median_stacking(frames_dir, sat_sigma)

    output_filename = f"./satstacker_output_{current_time}.tif"

    success = cv2.imwrite(output_filename, result_image)

    if success:
        print(f"#     Successfully saved 16-bit image to: {output_filename}")
    else:
        print("#     Failed to save the image.")
