from typing import Any, Callable, List, Dict
from utils import Encoder, Timer
from sprites import Car
from stable_baselines3 import PPO

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

encoder = Encoder(WIDTH, HEIGHT)
model = PPO.load("ppo_racer")

def game_loop(socketio, callback: Callable[[List[Dict[str, Any]]], None]):
    car = Car(1.5, 1, 7)
    timer = Timer(60, socketio)
    while True:
        encoder.background_image("background_with_track.png")
        action = action_to_keys[model.predict(car.get_state())[0]]
        car.update(*action)
        encoder.image("car.png", car.x, car.y, car.width, car.height, round(car.rotation))
        callback(encoder.data)
        encoder.clear()
        timer.tick()
