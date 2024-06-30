import os
import subprocess

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageEnhance


# 图像增强函数
def img_enhance(image, brightness, color, contrast, sharpness):
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    if brightness:
        image = enh_bri.enhance(brightness)
    # 色度增强
    enh_col = ImageEnhance.Color(image)
    if color:
        image = enh_col.enhance(color)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    if contrast:
        image = enh_con.enhance(contrast)
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    if sharpness:
        image = enh_sha.enhance(sharpness)
    return image


# 视频增强函数
def enhance_video(vid_path, output_path):
    # 打开视频文件
    src_video = cv2.VideoCapture(vid_path)
    fps = int(src_video.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率（每秒的帧数）
    width = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧的宽度
    height = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧的高度
    total_frame = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 遍历视频的所有帧
    for i in tqdm(range(total_frame)):  # 使用 tqdm 显示进度条
        success, frame = src_video.read()  # 读取视频的一帧
        if not success:
            break
        # # 去除水印
        # if watermark_area:
        #     x, y, w, h = watermark_area
        #     frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (15, 15), 0)
        # 转换为 PIL 图像
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 对图像进行增强
        enhanced_frame_pil = img_enhance(frame_pil, brightness=1.2, color=1.2, contrast=1.3, sharpness=1.3)
        # 转换回 OpenCV 图像
        enhanced_frame = cv2.cvtColor(np.array(enhanced_frame_pil), cv2.COLOR_RGB2BGR)
        # 写入增强后的帧到输出视频
        out_video.write(enhanced_frame)

    src_video.release()  # 释放视频文件
    out_video.release()  # 释放输出视频文件

    print(f"视频增强完成，结果已保存至 {output_path}")

# # 从视频中提取字幕的函数
# def extract_subtitles_from_video(vid_path):
#     # 初始化 PaddleOCR
#     ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, rec_char_type='ch')
#
#     # 打开视频文件
#     src_video = cv2.VideoCapture(vid_path)
#     fps = int(src_video.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率（每秒的帧数）
#     total_frame = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
#     save_text = []  # 用于存储提取的字幕文本
#
#     # 设置预览窗口
#     raw_title = '1'
#     cv2.startWindowThread()
#     cv2.namedWindow(raw_title)
#
#     # 遍历视频的所有帧
#     for i in tqdm(range(total_frame)):  # 使用 tqdm 显示进度条
#         success, frame = src_video.read()  # 读取视频的一帧
#         if i % fps == 0 and success:  # 每隔一秒处理一帧
#             # 只抽取下半部分图片
#             cropped_frame = frame[-50:-20, :]
#             # 转换为 PIL 图像
#             cropped_frame_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#             # 对图像进行增强
#             enhanced_frame_pil = img_enhance(cropped_frame_pil, brightness=1.2, color=1.2, contrast=1.3, sharpness=1.3)
#
#             # 转换回 OpenCV 图像
#             enhanced_frame = cv2.cvtColor(np.array(enhanced_frame_pil), cv2.COLOR_RGB2BGR)
#
#             # 预览原始和增强后的图像
#             tmp_img = enhanced_frame.copy()
#             cv2.putText(tmp_img, f'Frame: {i // fps}', (5, 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             cv2.imshow(raw_title, cropped_frame)
#             cv2.imshow(raw_title, tmp_img)
#             cv2.waitKey(50)  # 设置显示每帧的时间间隔
#
#             result = ocr.ocr(enhanced_frame, cls=True)  # 使用 OCR 识别字幕
#
#             # 检查OCR结果是否为空
#             if result and len(result) > 0:
#                 text_lines = result[0]  # 获取识别结果的第一行
#                 if text_lines and len(text_lines) > 0:
#                     res = text_lines[0][1][0]  # 提取识别的文本内容
#                     start_time = i // fps  # 计算当前帧对应的时间（秒）
#                     if [start_time, res] not in save_text:  # 避免重复添加相同的字幕
#                         save_text.append([start_time, res])  # 将时间和文本保存到列表
#
#     # 关闭预览窗口
#     cv2.destroyWindow(raw_title)
#     cv2.destroyAllWindows()
#
#     src_video.release()  # 释放视频文件
#
#     # 保存字幕结果到文本文件
#     subtitle_path = vid_path.replace('.mp4', '.txt')  # 生成字幕文件路径
#     print(f"字幕提取完成，结果已保存至 {subtitle_path}")
#     with open(subtitle_path, 'w', encoding='utf-8') as f:
#         previous_text = ""  # 用于存储前一行文本
#         for text in save_text:
#             current_text = text[1].strip()  # 去掉文本两端的空格
#             if current_text != previous_text:  # 检查当前文本是否与前一行不同
#                 f.write(f"{text[0]}: {current_text}\n")  # 写入时间和文本
#                 previous_text = current_text  # 更新前一行文本
#
#     return subtitle_path
