import eventlet

eventlet.monkey_patch()
import os
from multiprocessing import Process, Value

from flask import Flask, request, send_file, send_from_directory
from flask_socketio import SocketIO

from game import game_process

DIRNAME = os.path.dirname(__file__)
ALLOWED_EXTENSIONS = {"zip"}

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, message_queue="redis://", channel="socketio")
file_changed = Value('i', 0)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/images/<path:path>")
def send_image(path):
    return send_from_directory("images", path)


@socketio.on("connect")
def success_message():
    print("Connection Successful")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "File not found", 400
    file = request.files["file"]
    if file.filename == "":
        return "Empty file", 400
    if file and allowed_file(file.filename):
        file.save(
            os.path.join(DIRNAME, "flask_files/model.zip")
        )
        file_changed.value = 1
        # Return success message
        return "", 204
    # Unsupported media type
    return "File extension not allowed", 415


if __name__ == "__main__":
    process = Process(target=game_process, args=(file_changed,))
    process.start()
    socketio.run(app, host="0.0.0.0", port=8000, debug=False)
