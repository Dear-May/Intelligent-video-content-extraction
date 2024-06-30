import os

import cv2
import numpy as np


def video_target_tracking(video_path, top_left, bottom_right, output_path):
    # 读取视频
    cap = cv2.VideoCapture()
    cap.open(video_path)  # 打开视频文件
    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    base_dir, video_name_1 = os.path.split(output_path)
    video_name_1 = "1" + video_name_1
    output_path_1 = f"{base_dir}/{video_name_1}"

    # 获取视频的宽度和高度
    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))

    # 设置视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'h264')

    # 创建保存视频对象
    out = cv2.VideoWriter(output_path, fourcc, 25, (frame_w, frame_h))

    # 获取第一帧图像，并指定目标位置
    ret, frame = cap.read()

    # 目标位置 (行, 高, 列, 宽)
    r = top_left[1]
    h = bottom_right[1] - top_left[1]
    c = top_left[0]
    w = bottom_right[0] - top_left[0]
    window = (c, r, w, h)  # 定义窗口位置

    # 指定目标的感兴趣区域（ROI）
    roi = frame[r:r + h, c:c + w]

    # 计算直方图
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转换为HSV格式
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))  # 忽略低亮度的值
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])  # 计算直方图

    # 归一化
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 目标追踪
    # 设置窗口的终止条件: 最大迭代次数为10, 窗口中心漂移最小值为1
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        # 获取每一帧图像
        ret, frame = cap.read()
        if ret:
            # 计算直方图的反向投影
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 进行meanshift跟踪
            ret, window = cv2.meanShift(dst, window, term_crit)

            # 将追踪的位置绘制在视频上，并进行显示
            x, y, w, h = window
            print(f'位置:{window[:2]}')  # 打印窗口位置
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形
            out.write(frame)  # 写入输出视频
        else:
            break

    cap.release()
    out.release()
