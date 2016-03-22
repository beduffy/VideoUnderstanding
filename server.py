__author__ = 'Ben'

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, send_file
import main
from flask_socketio import SocketIO, send, emit

app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
app.debug = True

# landing page
@app.route("/")
def hello():
    return send_from_directory('static', 'landing_page.html')
    # return "index path eventually???" #TODO

# get static file
@app.route('/<path:path>')
def send_static_file(path):
    print 'static file', path
    return send_from_directory('static', path)

# download video
@app.route('/download_video/', methods=['POST'])
def download_video():
    data = dict((key, request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form.getlist(key)[0]) for key in request.form.keys())
    print 'data: ', data

    url = data.get('url')
    print 'url is: ', url

    main.pytube_download_and_info(url)
    return url

# process video
@app.route('/process_video/<path:path>')
def process_video(path):
    # print 'static file', path
    return send_from_directory('static', path)


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

# ALL FLASH SOCKETS BELOW

def ack():
    print 'message was received!'

@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)
    send(message)

@socketio.on('json')
def handle_json(json):
    print('received json: ' + str(json))
    send(json, json=True)

@socketio.on('my event')
def handle_my_custom_event(json):
    print('received json: ' + str(json))

    send_print_event('hey hey hey')

    # socketio.emit('some event', {'data': 42})

# @socketio.on('my event')
# def handle_my_custom_event(arg1, arg2, arg3):
#     print('received args: ' + arg1 + arg2 + arg3)
#     emit('my response', json, callback=ack)

# todo understnad below sends client info. broadcast=true works too
def send_print_event(json_data):
    print 'sending:', json_data
    socketio.emit('print_event', json_data)

@socketio.on('connect')
def test_connect():
    print('Client connected')
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on_error_default       # Handles the default namespace
def error_handler(e):
    print 'INSIDE ERROR HANDLER'
    print(request.event["message"]) # "my error event"
    print(request.event["args"])    # (data,)


#     TODO UNDERSTAND BELOW EXAMPLE
# @socketio.on('my event')
# def handle_my_custom_event(json):
#     print('received json: ' + str(json))
#     return 'one', 2

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    # app.run()


    print 'running socketio on app'
    socketio.run(app)


    #tpdp code doesnt get here.
    url_for('static', filename='video_results.css')
    url_for('static', filename='jquery.mousewheel.js')
    url_for('static', filename='handlebars-v4.0.5.js')

    send_print_event('hey hey hey')