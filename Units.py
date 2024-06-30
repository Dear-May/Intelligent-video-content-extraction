import os
import subprocess
import json


def check_and_modify_video_codec(filepath):
    try:
        # 运行 FFmpeg probe 命令
        command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name', '-of', 'json', filepath]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"Error in probing video: {result.stderr}")
            return None

        # 解析结果
        probe = json.loads(result.stdout)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_name'].startswith('h264')), None)

        # 检查是否为h264
        if not video_stream:
            base, ext = os.path.splitext(filepath)
            new_file_name = base + '_old' + ext
            os.rename(filepath, new_file_name)

            command_no_rect_h264 = f'ffmpeg -y -i {new_file_name} -vcodec h264 {filepath}'
            subprocess.run(command_no_rect_h264, shell=True)
            os.remove(new_file_name)

            return new_file_name
        return filepath
    except Exception as e:
        print(f"General error: {e}")
        return None
