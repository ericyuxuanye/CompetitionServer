from flask import Flask, send_file, send_from_directory
from flask_socketio import SocketIO
from game import game_loop
from time import time

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route("/")
def index():
    return send_file("index.html")


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images', path)


@socketio.on("connect")
def success_message():
    print("Connection Successful")

def on_game_frame(data):
    socketio.emit("frame", {"data": data})


if __name__ == '__main__':
    socketio.start_background_task(game_loop, socketio, on_game_frame)
    socketio.run(app, host="0.0.0.0", port=8888, debug=False)
