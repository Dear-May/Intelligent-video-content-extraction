from flask import Flask, render_template, request, Response

app = Flask(__name__)


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
    app.run()
