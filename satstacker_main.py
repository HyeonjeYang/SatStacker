# Copyright (c) 2026 Penta0308
import glob
import os
import shutil
from datetime import datetime
import subprocess

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

def genworkdir() -> str:
    print("# 1   Creating Job Directory.")
    work_dir = f"./satstacker_job_{current_time}/"
    os.makedirs(work_dir, exist_ok=True)
    print("# 1   Succeeded.")
    return work_dir

def takevideoandmaketrf(work_dir: str, time_start: str, time_dur: str, inputf_name:str) -> str:
    print("# 2   Extracting Vibration.")
    trf_path = os.path.join(work_dir, "transforms.trf")

    subprocess.run([
        'ffmpeg',
        '-y',
        '-v', 'error',
        '-ss', time_start, '-t', time_dur, '-i', inputf_name,
        '-vf', f'vidstabdetect=tripod=1:result={trf_path}:fileformat=ascii',
        '-f', 'null', '-'], check=True)

    print("# 2   Succeeded.")
    return trf_path

def takevideoanddeshakeandexportimage(work_dir: str, time_start: str, time_dur: str, inputf_name:str, trf_path:str) -> str:
    print("# 3   Processing Frames.")
    frames_dir = f"{work_dir}frames/"
    os.makedirs(frames_dir, exist_ok=True)

    image_out_pattern = os.path.join(frames_dir, "frame_%04d.tif")
    subprocess.run([
        'ffmpeg', '-y',
        '-v', 'error',
        '-ss', time_start, '-t', time_dur, '-i', inputf_name,
        '-vf', (
            f"negate,vidstabtransform=optzoom=0:tripod=1:crop=black:input={trf_path},negate,format=rgb48"
        ),
        '-pix_fmt', 'rgb48le',
        image_out_pattern], check=True)

    print("# 3   Succeeded.")
    return frames_dir

def cleanup_workspace(work_dir: str):
    print("# 5   Cleaning up temporary files.")
    try:
        shutil.rmtree(work_dir)
        print("# 5   Succeeded.")
    except Exception as e:
        print(f"# 5   Failed to clean up workspace: {e}")

if __name__ == "__main__":
    inputf_name = input("?     Drag video file here [path]: ").strip().replace("'", "").replace('"', "")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    work_dir = genworkdir()

    time_start = input("?     Enter the start time [s or mm:ss]: ")
    time_dur = input("?     Enter the duration [s]: ")

    trf_path = takevideoandmaketrf(work_dir, time_start, time_dur, inputf_name)

    # trf_path parameter
    frames_dir = takevideoanddeshakeandexportimage(work_dir, time_start, time_dur, inputf_name, trf_path)

    sat_sigma = input("?     Threshold for Satellite detection. Negative for darker than background(moon, etc.). Default -3.0 [float] SD: ")
    if sat_sigma == "":
        sat_sigma = -3.0
    else:
        sat_sigma = float(sat_sigma)

    result_image = lab_median_stacking(frames_dir, sat_sigma)

    output_filename = f"./satstacker_output_{current_time}.tif"

    success = cv2.imwrite(output_filename, result_image)

    if success:
        print(f"#     Successfully saved 16-bit image to: {output_filename}")
        cleanup_workspace(work_dir) # cleanup for memory!
    else:
        print("#     Failed to save the image. Temporary files are kept for debugging.")
