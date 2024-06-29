import cv2
import os
import numpy as np
import subprocess
from image_background import ensure_directory_exists

def preprocess_image(image, save_folder, frame_idx):
    """
    图像预处理，包含灰度转换、二值化、边缘检测和形态学操作
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    save_image_steps(gray, save_folder, 'gray', frame_idx)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
    save_image_steps(binary, save_folder, 'binary', frame_idx)

    edges = cv2.Canny(binary, 50, 150)  # 边缘检测
    save_image_steps(edges, save_folder, 'edges', frame_idx)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # 形态学闭操作
    morph = cv2.dilate(morph, kernel, iterations=1)  # 膨胀操作
    save_image_steps(morph, save_folder, 'morph', frame_idx)

    return morph

def save_image_steps(image, folder, step_name, frame_idx):
    """
    保存图像处理的每一步骤
    """
    step_folder = os.path.join(folder, step_name)
    ensure_directory_exists(step_folder)
    filename = os.path.join(step_folder, f"{step_name}_frame_{frame_idx}.jpg")
    cv2.imwrite(filename, image)

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

    whole_result_folder = os.path.join(save_folder, 'whole_result')
    ensure_directory_exists(whole_result_folder)
    cv2.imwrite(os.path.join(whole_result_folder, f'frame_{frame_idx}.jpg'), result)

    return result

def remove_watermark_from_video(input_video_path, top_left, bottom_right):
    """
    从视频中去除水印
    """
    video_filename = os.path.basename(input_video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    output_dir = os.path.join('uploads', video_name)
    ensure_directory_exists(output_dir)
    output_with_rect_path = os.path.join(output_dir, f"{video_name}_with_rect{video_ext}")
    output_no_rect_path = os.path.join(output_dir, f"{video_name}_no_rect{video_ext}")

    # 提取音频
    audio_path = os.path.join(output_dir, f"{video_name}.mp3")
    command = f"ffmpeg -i {input_video_path} -q:a 0 -map a {audio_path}"
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
    out_with_rect = cv2.VideoWriter(output_with_rect_path, fourcc, fps, (frame_width, frame_height))
    out_no_rect = cv2.VideoWriter(output_no_rect_path, fourcc, fps, (frame_width, frame_height))

    original_folder = os.path.join(output_dir, 'original')
    ensure_directory_exists(original_folder)

    result_folder = os.path.join(output_dir, 'result')
    ensure_directory_exists(result_folder)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 保存原图像
        original_image_path = os.path.join(original_folder, f'frame_{frame_idx}.jpg')
        cv2.imwrite(original_image_path, frame)

        frame_without_watermark = remove_watermark(frame, top_left, bottom_right, output_dir, frame_idx)

        mask = np.zeros_like(frame)
        mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (0, 0, 0)
        frame_with_mask = cv2.addWeighted(frame, 1, mask, 1, 0)

        replacement_region = frame_without_watermark[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        frame_with_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = replacement_region

        # 保存最终去水印图像到 result 文件夹中
        final_result_path = os.path.join(result_folder, f'frame_{frame_idx}.jpg')
        cv2.imwrite(final_result_path, frame_with_mask)

        # 在每一帧上绘制矩形框，并写入视频文件
        out_no_rect.write(frame_with_mask)
        frame_with_rect = frame_with_mask.copy()
        cv2.rectangle(frame_with_rect, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 2)
        out_with_rect.write(frame_with_rect)

    cap.release()
    # 释放 VideoWriter 对象
    out_with_rect.release()
    out_no_rect.release()

    # 合并音频和处理后的视频
    final_output_with_rect = os.path.join(output_dir, f"{video_name}_with_rect_with_audio{video_ext}")
    final_output_no_rect = os.path.join(output_dir, f"{video_name}_no_rect_with_audio{video_ext}")

    # 使用 ffmpeg 将原始音频合并到去水印的视频中
    command_with_rect = f"ffmpeg -i {output_with_rect_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {final_output_with_rect}"
    command_no_rect = f"ffmpeg -i {output_no_rect_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {final_output_no_rect}"
    subprocess.run(command_with_rect, shell=True)
    subprocess.run(command_no_rect, shell=True)

    print("去除水印完成，结果保留为", final_output_with_rect, "和", final_output_no_rect)
