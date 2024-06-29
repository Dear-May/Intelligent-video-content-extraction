import os
import cv2
import pytesseract
import re
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image, save_folder, frame_idx):
    """
    图像预处理，色彩增强
    """
    # 创建各步骤的文件夹
    cropped_folder = os.path.join(save_folder, 'cropped')
    ensure_directory_exists(cropped_folder)
    black_cropped_folder = os.path.join(save_folder, 'black_cropped')
    ensure_directory_exists(black_cropped_folder)
    hsv_folder = os.path.join(save_folder, 'hsv')
    ensure_directory_exists(hsv_folder)
    enhanced_folder = os.path.join(save_folder, 'enhanced')
    ensure_directory_exists(enhanced_folder)

    # 保存裁剪后的图像
    cv2.imwrite(os.path.join(cropped_folder, f'frame_{frame_idx}.jpg'), image)

    height, width, _ = image.shape
    black_top_height = int(0.35 * height)
    image_cropped = image.copy()
    image_cropped[:black_top_height, :] = 0

    # 保存裁剪后的黑化处理的图像
    cv2.imwrite(os.path.join(black_cropped_folder, f'frame_{frame_idx}.jpg'), image_cropped)

    # 保存HSV图像
    hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(hsv_folder, f'frame_{frame_idx}.jpg'), hsv)

    # 保存增强后的图像
    h, s, v = cv2.split(hsv)
    s = cv2.equalizeHist(s)
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(enhanced_folder, f'frame_{frame_idx}.jpg'), enhanced_image)

    return enhanced_image

def process_frame_with_tesseract(image):
    """
    使用Tesseract进行OCR处理，识别图像中的文本
    """
    config = '--psm 7 -l chi_sim'
    ocr_str = pytesseract.image_to_string(image, config=config).strip()
    return ocr_str

def is_blank_image(image, threshold=1):
    """
    判断图像是否为空白图像，通过检测白色像素比例
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(binary) / binary.size
    return white_ratio > threshold

def process_video(video_path):
    """
    处理视频，识别字幕并保存字幕和对应的图像帧
    """
    v = cv2.VideoCapture(video_path)
    fps = v.get(cv2.CAP_PROP_FPS)
    total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

    seventh_height = height // 7
    subtitle_top = height - seventh_height

    video_name = os.path.basename(video_path).split('.')[0]
    save_folder = os.path.join('uploads', video_name)
    ensure_directory_exists(save_folder)

    rect_folder = os.path.join(save_folder, 'rect')
    ensure_directory_exists(rect_folder)

    subtitles = []
    last_ocr_str = ""
    last_frame = None

    def process_frame(idx):
        """
        处理单帧图像，进行字幕识别和过滤
        """
        nonlocal last_ocr_str, last_frame
        v.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = v.read()
        if not ret:
            return

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        original_folder = os.path.join(save_folder, 'original')
        ensure_directory_exists(original_folder)

        # 保存原图像（裁切前）
        original_image_path = os.path.join(original_folder, f'frame_{idx}.jpg')
        cv2.imwrite(original_image_path, frame)

        bottom_seventh = frame[subtitle_top:, :]
        processed_frame = preprocess_image(bottom_seventh, save_folder, idx)

        if is_blank_image(processed_frame):
            print(f"Frame {idx}: Blank image detected, skipping.")
            return

        ocr_str = process_frame_with_tesseract(processed_frame)
        ocr_str = re.sub(r'[^\u4e00-\u9fa5]', '', ocr_str)  # 只保留中文字符

        if ocr_str:  # 只有在识别到字幕不为空时才进行处理
            print(f"Frame {idx}: OCR Result: {ocr_str}")

            if fuzz.ratio(ocr_str, last_ocr_str) < 90:  # 与上一句字幕进行相似性对比，若相似度低于90%则保存
                is_similar = any(fuzz.ratio(ocr_str, subtitle[1]) > 70 for subtitle in subtitles)
                if not is_similar:
                    last_ocr_str = ocr_str
                    last_frame = processed_frame
                    timestamp = idx / fps
                    subtitles.append((timestamp, ocr_str))

                    frame_path = os.path.join(rect_folder, f'frame_{idx}.jpg')
                    cv2.imwrite(frame_path, processed_frame)

                    return (timestamp, ocr_str)

    # 使用 ThreadPoolExecutor 来并行处理视频帧
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 提交任务到线程池，每个任务调用 process_frame 函数来处理特定帧
        futures = [executor.submit(process_frame, idx) for idx in range(0, total_frames, int(fps / 2))]
        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()

    # 创建字幕文件的路径
    subtitles_path = os.path.join(save_folder, f'{video_name}_subtitles.txt')
    # 以写入模式打开字幕文件，并指定编码为 UTF-8
    with open(subtitles_path, 'w', encoding='utf-8') as txt_file:
        # 按时间戳排序字幕列表，并写入文件
        for timestamp, subtitle in sorted(subtitles):
            if subtitle:
                # 写入时间戳和字幕内容到文件
                txt_file.write(f"{timestamp:.2f}: {subtitle}\n")
                # 打印正在写入的内容到控制台
                print(f"Writing to file: {timestamp:.2f}: {subtitle}")

    return subtitles_path
