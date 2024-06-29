import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QColorDialog
from watermark_removal import remove_watermark_from_video
from video_subtitle import process_video
from image_background import change_background


def main():
    while True:
        print("请选择一个功能:")
        print("1. 去除视频水印")
        print("2. 识别视频字幕")
        print("按其他键退出")

        choice = input("输入序号 (1/2): ").strip()

        if choice == "1":
            input_video_path = input("输入视频路径: ").strip()
            top_left_x = int(input("输入左上角X坐标: ").strip())
            top_left_y = int(input("输入左上角Y坐标: ").strip())
            bottom_right_x = int(input("输入右下角X坐标: ").strip())
            bottom_right_y = int(input("输入右下角Y坐标: ").strip())
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            remove_watermark_from_video(input_video_path, top_left, bottom_right)

        elif choice == "2":
            input_video_path = input("输入视频路径: ").strip()
            process_video(input_video_path)

        else:
            print("退出程序")
            break


if __name__ == "__main__":
    main()
