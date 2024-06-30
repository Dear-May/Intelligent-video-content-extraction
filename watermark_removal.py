import cv2
import os
import numpy as np
import subprocess


def preprocess_image(image, save_folder, frame_idx):
    """
    图像预处理，包含灰度转换、二值化、边缘检测和形态学操作
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
    edges = cv2.Canny(binary, 50, 150)  # 边缘检测
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # 形态学闭操作
    morph = cv2.dilate(morph, kernel, iterations=1)  # 膨胀操作

    return morph


def remove_watermark(image, top_left, bottom_right, save_folder, frame_idx):
    """
    去除图像水印
    """
    processed = preprocess_image(image, save_folder, frame_idx)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image[:, :, 0])
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
    result = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return result


def remove_watermark_from_video(input_video_path, top_left, bottom_right, filename, output_dir):
    """
    从视频中去除水印
    """
    video_filename = os.path.basename(input_video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    output_no_rect_path = os.path.join(output_dir, filename)

    # 提取音频
    audio_path = os.path.join(output_dir, f"{video_name}.mp3")
    command = f"ffmpeg -y -i {input_video_path} -q:a 0 -map a {audio_path}"
    subprocess.run(command, shell=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 创建 VideoWriter 对象，用于保存带矩形框和不带矩形框的视频
    out_no_rect = cv2.VideoWriter(output_no_rect_path, fourcc, fps, (frame_width, frame_height))

    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_without_watermark = remove_watermark(frame, top_left, bottom_right, output_dir, frame_idx)

        mask = np.zeros_like(frame)
        mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (0, 0, 0)
        frame_with_mask = cv2.addWeighted(frame, 1, mask, 1, 0)

        replacement_region = frame_without_watermark[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        frame_with_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = replacement_region

        # 在每一帧上绘制矩形框，并写入视频文件
        out_no_rect.write(frame_with_mask)
        frame_with_rect = frame_with_mask.copy()
        cv2.rectangle(frame_with_rect, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 2)

    cap.release()
    # 释放 VideoWriter 对象
    out_no_rect.release()

    # 合并音频和处理后的视频
    final_output_no_rect_1 = os.path.join(output_dir, "1" + filename)
    final_output_no_rect = os.path.join(output_dir, filename)

    # 使用 ffmpeg 将原始音频合并到去水印的视频中
    command_no_rect = f"ffmpeg -y -i {output_no_rect_path} -i {audio_path} -vcodec h264 -c:a aac -strict experimental {final_output_no_rect_1}"
    subprocess.run(command_no_rect, shell=True)
    command_no_rect_h264 = f'ffmpeg -y -i {final_output_no_rect_1} -vcodec h264 {final_output_no_rect}'
    subprocess.run(command_no_rect_h264, shell=True)

    os.remove(f"{output_dir}/1{filename}")
    os.remove(audio_path)

    print("去除水印完成，结果保留为", final_output_no_rect)
