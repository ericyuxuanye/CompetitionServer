import eventlet

eventlet.monkey_patch()
import os
from multiprocessing import Process
import redis

from flask import Flask, request, send_file, send_from_directory
from flask_socketio import SocketIO

from game import game_process

UPLOAD_FOLDER = "~/flask_files"
ALLOWED_EXTENSIONS = {"pt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, message_queue="redis://", channel="socketio")


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
            os.path.join(os.path.expanduser(app.config["UPLOAD_FOLDER"]), "model.pt")
        )
    # Return success message
    return "", 204


if __name__ == "__main__":
    process = Process(target=game_process)
    process.start()
    socketio.run(app, host="0.0.0.0", port=8000, debug=False)
