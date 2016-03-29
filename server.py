__author__ = 'Ben'

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, send_file
from flask_socketio import SocketIO, send, emit
# from flask.ext.socketio import SocketIO
import utilities.globals as globals
from utilities.globals import log, HEADER_SIZE
import thread
import main

app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
# app.debug = True
globals.init_globals(app)


# landing page
@app.route("/")
def hello():
    return send_from_directory('static', 'landing_page.html')

# get static file
@app.route('/<path:path>')
def send_static_file(path):
    print 'static file', path
    return send_from_directory('static', path)

# download video
@app.route('/download_video/', methods=['POST'])
def download_video():
    data = dict((key, request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form.getlist(key)[0]) for key in request.form.keys())
    url = data.get('url')

    log('Downloading youtube URL: ', url, header=HEADER_SIZE)

    # todo stop js from clicking twice.
    name = main.download_video(url)
    return name

# process video
@app.route('/process_video/', methods=['POST'])
def process_video():
    data = dict((key, request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form.getlist(key)[0]) for key in request.form.keys())

    current_dir = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(current_dir, 'example_images', data['name'], data['name'] + '.mp4') #TODO VERY CAREFUL HERE MIGHT NOT BE MP4

    thread.start_new_thread(main.process_video, (video_path, data['url']))

    # todo change
    return 'Video has completed processing'

# get image
@app.route('/example_images/<path:path>')
def send_image(path):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(current_dir, 'example_images', path)
    # print full_path

    return send_file(full_path, mimetype='image/png')

# get result_struct.json
@app.route("/get_json/<path:video_folder>", methods=['GET'])
def get_json_struct(video_folder):
    print 'video folder: ', video_folder
    current_dir = os.path.dirname(os.path.realpath(__file__))
    json_struct_path = os.path.join(current_dir, 'example_images', video_folder, 'metadata', 'result_struct.json')

    print 'json_struct_path: ', json_struct_path

    json_struct = {'images': []}
    if os.path.isfile(json_struct_path):
        with open(json_struct_path) as data_file:
            json_struct = json.load(data_file)

            print json_struct
            return jsonify(**json_struct)

    return 'No json struct found!'

# get all_videos json
@app.route("/get_all_videos", methods=['GET'])
def get_json_videos():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    json_videos_path = os.path.join(current_dir, 'example_images', 'all_videos.json')

    print 'json_struct_path: ', json_videos_path

    json_videos = {'videos': []}
    if os.path.isfile(json_videos_path):
        with open(json_videos_path) as data_file:
            json_videos = json.load(data_file)

            print json_videos
            return jsonify(**json_videos)

    return jsonify(**json_videos)

# ALL FLASH SOCKETS BELOW

def ack():
    print 'message was received!'

@globals.socketio.on('connect')
def test_connect():
    print('Client connected')
    # emit('print_event', {'s': 'Connected to server', 'header' : 6})
    log('Connected to server', header=4)

@globals.socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    print 'Running socketio on app'
    globals.socketio.run(app, debug=True, log_output=False)