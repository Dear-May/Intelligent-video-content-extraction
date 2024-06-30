import datetime
import os
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5

import cv2
import numpy as np
import pytesseract
import requests
from skimage.metrics import mean_squared_error

# %%
# 百度翻译API相关配置
appid = '20240602002068079'
appkey = 'MHRDSt8USx9LPmQJS7lN'
from_lang = 'auto'
endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path
center_x = None
center_y = None
text_height = None


# 相关工具函数
def create_and_clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def save_key_frames(k_frames, base_dir):
    recognition_dir = os.path.join(base_dir, "recognition")
    create_and_clear_folder(recognition_dir)  # 创建并清空 recognition 文件夹
    for idx, kf in enumerate(k_frames):
        img_name = f"{idx:03d}.jpg"
        img_path = os.path.join(recognition_dir, img_name)
        cv2.imwrite(img_path, kf['frame'])


# 读取SRT文件
def read_srt_file(srt_file_path):
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()

    srt_pattern = re.compile(
        r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)\s+(?=\d+\s+\d{2}:\d{2}:\d{2},'
        r'\d{3}\s*-->|$)',
        re.DOTALL
    )

    matches = srt_pattern.findall(srt_content)

    subtitles_info = []

    for match in matches:
        index, start_time, end_time, text = match
        subtitles_info.append({'序号': index, '开始时间': start_time, '结束时间': end_time, '字幕内容': text})

    # 输出调试信息
    for subtitle in subtitles_info:
        print(f"调试信息: 序号: {subtitle['序号']}, 开始时间: {subtitle['开始时间']}, 结束时间: {subtitle['结束时间']}")

    return subtitles_info


# MD5加密函数
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


def remove_readonly(func, path, excinfo):
    os.chmod(path, 0o777)
    func(path)


# %%
# 1.读取视频文件
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    # 打印视频信息
    print(f'视频路径   : {video_path:<20} | 帧数: {num_frames:<5} | 帧率: {fps:<5} | 视频尺寸: {width} x {height}')
    return cap, num_frames, fps, height, width


# %%
# 2.获取视频帧
def get_frame_index(time_str: str, fps: float):
    t = time_str.replace(',', '.').split(':')  # 将逗号替换为点号
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(
            'Time data "{}" does not match format "%H:%M:%S"'.format(time_str))
    index = int(td.total_seconds() * fps)
    return index


# %%
# 3.截取字幕区域
def extract_subtitle_area(frames, height, h_ratio_start=0.86, h_ratio_end=0.94):
    h1, h2 = int(height * h_ratio_start), int(height * h_ratio_end)
    processed_frames = []
    for idx, frame in enumerate(frames):
        # 截取字幕区域并转换为灰度图
        subtitle_frame = frame[h1:h2, :]
        gray = cv2.cvtColor(subtitle_frame, cv2.COLOR_BGR2GRAY)

        # 双边滤波降噪
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny边缘检测
        edges = cv2.Canny(filtered, 100, 200)

        # 形态学操作：膨胀文字，以便更好地连接文字的轮廓
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 轮廓检测
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 创建一个全黑的掩模
            mask = np.zeros_like(dilated)
            # 填充找到的轮廓，使文字变白
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            # 应用掩模获取最终的文字图像
            final_text_image = cv2.bitwise_and(filtered, filtered, mask=mask)
            processed_frames.append(final_text_image)
        else:
            processed_frames.append(np.zeros_like(dilated))  # 如果没有找到轮廓，返回一个空黑图

    return processed_frames


# %%
# 4. mobabn
def count_white_pixels(image):
    # 计算白色像素的数量
    return np.sum(image == 255)


def process_image(image):
    # 应用二值化处理，将比文字像素值小的部分全部设置为黑色
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # 创建结果图像，初始化为全黑
    result = np.zeros_like(image)
    result[binary == 255] = 255
    # 应用形态学操作，去除小的黑点
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return morph


def crop_to_text_area(image):
    image = process_image(image)
    # 应用二值化处理，识别白色区域
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # 使用形态学操作去噪点
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最小和最大边界
    min_x, min_y = image.shape[1], image.shape[0]
    max_x, max_y = 0, 0

    # 找到包含所有文字区域的最小矩形
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # 调整裁剪区域的边缘
    padding = 10  # 调整的像素数
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image.shape[1], max_x + padding)
    max_y = min(image.shape[0], max_y + padding)

    # 裁剪图像
    cropped_image = image[min_y:max_y, min_x:max_x]

    return cropped_image


def filter_similar_frames(frames, mse_threshold=700):
    k_frames = [
        {'start': 0, 'end': 0, 'frame': frames[0], 'text': '', 'white_pixel_count': count_white_pixels(frames[0])}
    ]

    for idx in range(1, len(frames)):
        mse = mean_squared_error(frames[idx - 1], frames[idx])
        white_pixel_count = count_white_pixels(frames[idx])
        if mse < mse_threshold:
            if white_pixel_count < k_frames[-1]['white_pixel_count']:
                k_frames[-1] = {
                    'start': k_frames[-1]['start'],
                    'end': idx,
                    'frame': frames[idx],
                    'text': '',
                    'white_pixel_count': white_pixel_count
                }
            else:
                k_frames[-1]['end'] = idx
        else:
            k_frames.append({
                'start': idx, 'end': idx,
                'frame': frames[idx],
                'text': '',
                'white_pixel_count': white_pixel_count
            })

    # 删除白色像素计数字段并裁剪图像，只保留文字区域
    for kf in k_frames:
        del kf['white_pixel_count']
        kf['frame'] = crop_to_text_area(kf['frame'])

    return k_frames


# %%
# 5. 识别字幕内容
def perform_ocr_on_frames(k_frames, lang='chi_sim'):
    config = '--psm 7'
    previous_text = None
    previous_start = None
    merged_frames = []

    for idx, kf in enumerate(k_frames):
        ocr_str = pytesseract.image_to_string(kf['frame'], lang=lang, config=config)
        ocr_str = ocr_str.strip().replace(' ', '')

        if ocr_str:
            if ocr_str == previous_text:
                # 合并字幕时间段
                merged_frames[-1]['end'] = kf['end']
            else:
                if previous_text:
                    merged_frames.append({'start': previous_start, 'end': kf['start'], 'text': previous_text})
                previous_text = ocr_str
                previous_start = kf['start']

    # 添加最后一个字幕段
    if previous_text:
        merged_frames.append({'start': previous_start, 'end': k_frames[-1]['end'], 'text': previous_text})

    # 移除没有文本的帧，并将合并后的结果赋值回 k_frames
    k_frames[:] = merged_frames

    # 打印合并后的结果
    for kf in k_frames:
        print(f"{kf['start']} --> {kf['end']} : {kf['text']}")


# %%
# 6. 格式化字幕
def get_srt_timestamp(frame_index: int, fps: float):
    td = datetime.timedelta(seconds=frame_index / fps)
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)


def generate_srt(k_frames, fps):
    srt_content = []
    for idx, kf in enumerate(k_frames):
        time1 = get_srt_timestamp(kf['start'], fps)
        time2 = get_srt_timestamp(kf['end'], fps)
        srt_content.append(f"{idx + 1}")
        srt_content.append(f"{time1} --> {time2}")
        srt_content.append(kf['text'])
        srt_content.append("")  # 空行
    return "\n".join(srt_content)


# %%
# 7.消除字幕区域中的文字
# 使用inpaint修复字幕区域
def remove_text_with_inpainting(frame, mask):
    inpainted_frame = cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)
    return inpainted_frame


def process_frame(frame, template, top_left, bottom_right):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_frame)
    cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return remove_text_with_inpainting(frame, mask)


# 模板匹配并处理字幕
def match_and_draw_boxes(video_path, base_dir, subtitles_info):
    global center_x, center_y, text_height
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognition_dir = os.path.join(base_dir, "recognition")

    output_video_path = os.path.join(base_dir, f"{video_name}_clear.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_buffers = []

    for subtitle in subtitles_info:
        frame_idx_start = get_frame_index(subtitle['开始时间'], fps)
        frame_idx_end = get_frame_index(subtitle['结束时间'], fps)
        subtitle_image_path = os.path.join(recognition_dir, f"{int(subtitle['序号']) - 1:03d}.jpg")
        template = cv2.imread(subtitle_image_path, 0)

        match_found = False
        top_left = None
        bottom_right = None

        for frame_idx in range(frame_idx_start, frame_idx_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if not match_found:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.6:
                    top_left = max_loc
                    h, w = template.shape
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    center_x, center_y = top_left[0] + w // 2, top_left[1] + h // 2
                    text_height = h + int(0.2 * h)
                    match_found = True

            if match_found:
                frame_buffers.append((frame, template, top_left, bottom_right))
            else:
                out.write(frame)

    cap.release()

    # 使用多线程进行并行处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(lambda x: process_frame(*x), frame_buffers)

    for frame in results:
        out.write(frame)

    out.release()


# %%
# 8. 调用百度翻译通用API进行翻译
def baiduTranslate(query, flag=1):
    if flag:
        to_lang = 'en'  # 译文语种
    else:
        to_lang = 'zh'  # 译文语种

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # 建立请求
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    try:
        # 发送请求
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()
        # 获取翻译结果
        return result['trans_result'][0]['dst']
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


# 生成SRT文件内容
def generate_translated_srt(subtitles_info, flag=1):
    srt_content = []
    for subtitle in subtitles_info:
        time1 = subtitle['开始时间']
        time2 = subtitle['结束时间']
        print("原文", subtitle['字幕内容'])
        translated_text = baiduTranslate(subtitle['字幕内容'], flag)
        if translated_text is None:
            translated_text = subtitle['字幕内容']  # 如果翻译失败，则使用原始文本
        print("译文", translated_text)
        srt_content.append(f"{subtitle['序号']}")
        srt_content.append(f"{time1} --> {time2}")
        srt_content.append(translated_text)
        srt_content.append("")  # 空行
        # 添加随机睡眠时间，避免请求频率超限
        time.sleep(random.uniform(0.8, 1.5))
    return "\n".join(srt_content)


# %%
# 9.将翻译后的字幕内容写入视频中
def write_subtitles_to_video(video_path, subtitles_info, output_path):
    global center_x, center_y, text_height

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return

    subtitle_idx = 0
    subtitle = subtitles_info[subtitle_idx] if subtitles_info else None

    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的时间戳
        current_time = frame_idx / fps

        # 更新字幕
        if subtitle and get_frame_index(subtitle['结束时间'], fps) < frame_idx:
            subtitle_idx += 1
            if subtitle_idx < len(subtitles_info):
                subtitle = subtitles_info[subtitle_idx]
            else:
                subtitle = None

        # 在当前帧上绘制字幕
        if subtitle and get_frame_index(subtitle['开始时间'], fps) <= frame_idx <= get_frame_index(subtitle['结束时间'],
                                                                                                   fps):
            text_size = text_height / 30  # 根据文字高度调整文字大小，可以调整这个比例
            wrapped_text = wrap_text(subtitle['字幕内容'], text_size, width - 20)  # 换行处理，宽度减去一些边距
            y0 = center_y - text_height // 2  # 从中心位置向上偏移半个高度

            for i, line in enumerate(wrapped_text):
                y = y0 + i * text_height  # 每行向下偏移一个文字高度
                text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)[0][0]
                x = center_x - text_width // 2  # 计算文本在水平居中的位置
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 3,
                            cv2.LINE_AA)  # 加粗效果

        out.write(frame)

    cap.release()
    out.release()


def wrap_text(text, text_size, max_width):
    words = text.split(' ')
    lines = []
    line = ''
    for word in words:
        if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)[0][0] < max_width:
            line += word + ' '
        else:
            lines.append(line.strip())
            line = word + ' '
    lines.append(line.strip())
    return lines


# %%
def video_subtitle_recognition(start_time, end_time, original_file_path, base_dir, file_name):
    # 读取视频
    video_path = original_file_path
    cap, num_frames, fps, height, width = read_video(video_path)
    # 确定起止时间
    time_start = start_time
    time_end = end_time
    ocr_start = get_frame_index(time_start, fps) if time_start else 0
    ocr_end = get_frame_index(time_end, fps) if time_end else num_frames
    num_ocr_frames = ocr_end - ocr_start
    center_x, center_y, text_height, = 0, 0, 0

    print(f'ocr_start       :  {ocr_start}\n'
          f'ocr_end         :  {ocr_end}\n'
          f'num_ocr_frames  :  {num_ocr_frames}')

    # 提取字幕区域
    cap.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
    frames = [cap.read()[1] for _ in range(ocr_end - ocr_start)]
    z_frames = extract_subtitle_area(frames, height)

    # 去除相似度较高的帧
    k_frames = filter_similar_frames(z_frames)
    save_key_frames(k_frames, base_dir)
    for kf in k_frames:
        print(f"{kf['start']} --> {kf['end']} : {kf['text']}")

    # 识别字幕内容
    perform_ocr_on_frames(k_frames)
    # 生成SRT字幕文件内容
    srt_content = generate_srt(k_frames, fps)
    # 生成SRT文件路径
    video_name = os.path.splitext(file_name)[0]
    srt_file_path = os.path.join(base_dir, f"{video_name}_primary.srt")
    # 输出SRT文件
    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    # 读取SRT文件
    srt_info = read_srt_file(srt_file_path)
    print(srt_info)

    # 根据字幕图片对视频做模板匹配并修复字幕
    match_and_draw_boxes(video_path, base_dir, srt_info)

    # 翻译字幕并生成新的SRT文件
    translated_srt_content = generate_translated_srt(srt_info)
    translated_srt_file_path = os.path.join(base_dir, f"{video_name}_translate.srt")
    with open(translated_srt_file_path, "w", encoding="utf-8") as translated_srt_file:
        translated_srt_file.write(translated_srt_content)

    # 读取翻译后的SRT文件
    translated_srt_info = read_srt_file(translated_srt_file_path)

    # 写翻译后的字幕到视频中
    output_video_path = os.path.join(base_dir, f"{video_name}_translated.mp4")
    write_subtitles_to_video(f"{base_dir}/{video_name}_clear.mp4", translated_srt_info, output_video_path)

    final_output_no_rect_1 = f"{base_dir}/" + f"{video_name}_translated.mp4"
    final_output_no_rect = f"{base_dir}/" + f"translated_{video_name}.mp4"

    command_no_rect_h264 = f'ffmpeg -y -i {final_output_no_rect_1} -vcodec h264 {final_output_no_rect}'
    subprocess.run(command_no_rect_h264, shell=True)

    os.remove(f"{base_dir}/{video_name}_clear.mp4")
    os.remove(f"{base_dir}/{video_name}_primary.srt")
    os.remove(f"{base_dir}/{video_name}_translate.srt")
    os.remove(f"{base_dir}/{video_name}_translated.mp4")

    folder_path = f"{base_dir}/recognition"
    try:
        shutil.rmtree(folder_path, onerror=remove_readonly)
        print(f"Successfully deleted the folder: {folder_path}")
    except PermissionError as e:
        print(f"Failed to delete the folder: {folder_path}. PermissionError: {e}")
    except Exception as e:
        print(f"Failed to delete the folder: {folder_path}. Error: {e}")

    print(f"字幕识别及翻译完成，输出文件：{final_output_no_rect}")
