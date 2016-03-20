__author__ = 'Ben'

import sys
import BaseHTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import os

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

# HandlerClass = SimpleHTTPRequestHandler
# ServerClass  = BaseHTTPServer.HTTPServer
# Protocol     = "HTTP/1.0"
#
# if sys.argv[1:]:
#     port = int(sys.argv[1])
# else:
#     port = 8000
# server_address = ('127.0.0.1', port)
#
# HandlerClass.protocol_version = Protocol
# httpd = ServerClass(server_address, HandlerClass)
#
# sa = httpd.socket.getsockname()
# print "Serving HTTP on", sa[0], "port", sa[1], "..."
# httpd.serve_forever()


app = Flask(__name__, static_url_path='')
app.debug = True

@app.route("/")
def hello():
    return send_from_directory('', 'video_results.html')
    return "Hello World!!!!!"

@app.route("/<path:path>")
def send_css(path):

    return send_from_directory('', path)

# search route
@app.route('/search', methods=['POST'])
def search():

    if request.method == "POST":

        RESULTS_ARRAY = []

        # get url
        image_url = request.form.get('img')

        try:

            # initialize the image descriptor
            cd = ColorDescriptor((8, 12, 3))

            # load the query image and describe it
            from skimage import io
            import cv2
            query = io.imread(image_url)
            query = (query * 255).astype("uint8")
            (r, g, b) = cv2.split(query)
            query = cv2.merge([b, g, r])
            features = cd.describe(query)

            # perform the search
            searcher = Searcher(INDEX)
            results = searcher.search(features)

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})

            # return success
            return jsonify(results=(RESULTS_ARRAY[:3]))

        except:

            # return error
            jsonify({"sorry": "Sorry, no results! Please try again."}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    url_for('static', filename='video_results.css')
    url_for('static', filename='jquery.mousewheel.js')
    url_for('static', filename='handlebars-v4.0.5.js')