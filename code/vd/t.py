import cv2
import numpy as np

# 读取第一个视频
cap1 = cv2.VideoCapture()
if not cap1.open('./video/d5.mp4'):
    print("无法打开第一个视频文件")
    exit()

# 读取第二个视频
cap2 = cv2.VideoCapture()
if not cap2.open('d5_enhanced.mp4'):
    print("无法打开第二个视频文件")
    cap1.release()
    exit()

# 缩小比例 (例如 0.8 表示缩小一半)
scale_factor = 0.8

while True:
    # 读取第一个视频的帧
    ret1, frame1 = cap1.read()
    # 读取第二个视频的帧
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # 缩小帧的大小
        frame1_resized = cv2.resize(frame1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # 如果两个视频帧的大小不一样，则调整大小
        if frame1_resized.shape != frame2_resized.shape:
            frame2_resized = cv2.resize(frame2_resized, (frame1_resized.shape[1], frame1_resized.shape[0]))

        # 将两个视频帧水平拼接在一起
        combined_frame = np.hstack((frame1_resized, frame2_resized))

        # 显示组合帧
        cv2.imshow('Combined Video', combined_frame)
    elif ret1:
        # 只显示第一个视频帧
        frame1_resized = cv2.resize(frame1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow('Combined Video', frame1_resized)
    elif ret2:
        # 只显示第二个视频帧
        frame2_resized = cv2.resize(frame2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow('Combined Video', frame2_resized)
    else:
        # 如果两个视频都结束，退出循环
        break

    # 检查按键，按ESC键退出
    if cv2.waitKey(25) & 0xFF == 27:
        break

# 释放视频对象
cap1.release()
cap2.release()
cv2.destroyAllWindows()
