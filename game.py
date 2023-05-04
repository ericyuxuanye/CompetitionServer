from multiprocessing import Process
from typing import Any, Callable
from flask_socketio import SocketIO

from stable_baselines3 import PPO

from sprites import Car
from utils import Encoder, Timer

action_to_keys = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
]

WIDTH = 1024
HEIGHT = 768

def game_process():
    socketio = SocketIO(message_queue="redis://", channel="socketio")
    encoder = Encoder(WIDTH, HEIGHT)
    model = PPO.load("ppo_racer")
    car = Car(1.5, 1, 7)
    timer = Timer(60)
    while True:
        encoder.background_image("background_with_track.png")
        action = action_to_keys[model.predict(car.get_state())[0]]
        car.update(*action)
        encoder.image(
            "car.png", car.x, car.y, car.width, car.height, round(car.rotation)
        )
        socketio.emit("frame", {"data": encoder.data})
        encoder.clear()
        timer.tick()
