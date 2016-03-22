__author__ = 'Ben'

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, send_file
import main

app = Flask(__name__, static_url_path='')
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
    # print 'static file', path
    print 'heyaaaaaaaa'

    # url = request.form
    # print 'data:', request.data
    # print 'form', request.form
    # print 'args ', request.args
    data = dict((key, request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form.getlist(key)[0]) for key in request.form.keys())
    print 'data: ', data

    url = data.get('url')
    print 'url is: ', url
    # url = request.form.get('url')
    # url = request.args.get("url")
    # url = json.loads(request.data)
    # url = request.get_json()
    # print 'URL received by server: ', url


    main.pytube_download_and_info(url)
    # return send_from_directory('static', path)
    # return 'hey'
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

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run()
    url_for('static', filename='video_results.css')
    url_for('static', filename='jquery.mousewheel.js')
    url_for('static', filename='handlebars-v4.0.5.js')