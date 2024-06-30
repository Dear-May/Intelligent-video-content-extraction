import os
import re

from flask import Flask, render_template, request, jsonify

from video_subtitle_recognition import video_subtitle_recognition
from watermark_removal import remove_watermark_from_video

app = Flask(__name__)

# 配置上传文件夹路径
UPLOAD_FOLDER = 'static/video/upload_video'
PROCESSED_FOLDER = 'static/video/processed_video'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route('/video_translation', methods=['POST'])
def video_translation():
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    file_name = data.get('videoUrl')
    match = re.search(r'/([^/]+\.[a-zA-Z0-9]+)$', file_name)
    if match:
        file_name = match.group(1)

    original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    translated_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f'translated_{file_name}')

    video_subtitle_recognition(start_time, end_time, original_file_path, app.config['PROCESSED_FOLDER'], file_name)
    print('success')
    return jsonify({'fileUrl': translated_file_path}), 200


@app.route('/remove_watermark', methods=['POST'])
def remove_watermark():
    data = request.get_json()
    left_top_x = data.get('leftTopX')
    left_top_y = data.get('leftTopY')
    right_bottom_x = data.get('rightBottomX')
    right_bottom_y = data.get('rightBottomY')
    file_name = data.get('videoUrl')
    match = re.search(r'/([^/]+\.[a-zA-Z0-9]+)$', file_name)
    if match:
        file_name = match.group(1)

    original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f'watermark_removed_{file_name}')
    finally_file_name = f'watermark_removed_{file_name}'
    top_left = (left_top_x, left_top_y)
    bottom_right = (right_bottom_x, right_bottom_y)
    remove_watermark_from_video(original_file_path, top_left, bottom_right, finally_file_name,
                                app.config['PROCESSED_FOLDER'])
    print('success')
    return jsonify({'fileUrl': processed_file_path}), 200


@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 保存文件到指定路径
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # 返回相对路径
        file_url = os.path.join('/static/video/upload_video', file.filename)
        return jsonify({'fileUrl': file_url}), 200

    return jsonify({'error': 'File upload failed'}), 500


@app.route('/WatermarkRemoval')
def WatermarkRemoval():
    return render_template('WatermarkRemoval.html')


@app.route('/subtitling')
def subtitling():
    return render_template('subtitling.html')


@app.route('/ImageEnhancement')
def ImageEnhancement():
    return render_template('ImageEnhancement.html')


@app.route('/VideoTargetTracking')
def VideoTargetTracking():
    return render_template('VideoTargetTracking.html')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
