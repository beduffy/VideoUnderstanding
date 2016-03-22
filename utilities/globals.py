# from flask import Flask
from flask_socketio import SocketIO
# s = None

class SocketPrint:
    socketio = None

    def __init__(self, socket):
        socketio = socket

    def s_print(self, str):
        print 'calling s_print'
        print str
        socketio.emit('print_event', str)

def init_globals(app):
    # global app = Flask(__name__, static_url_path='')
    # app.config['SECRET_KEY'] = 'secret!'
    # app.debug = True
    global socketio
    socketio = SocketIO(app)
    global s
    s = SocketPrint(socketio)

def log(*args):
    print 'inside log'
    # todo might not need below 3 lines can just pass args to join?
    str_list = []
    count = 0

    print 'length: ', len(args)

    for i in args:
        print count, str(i)
        str_list.append(str(i))

        count += 1

    ret_str = ' '
    ret_str = ret_str.join(str_list)

    print ret_str
    socketio.emit('print_event', ret_str)
    # print str
    # socketio.emit('print_event', str)


