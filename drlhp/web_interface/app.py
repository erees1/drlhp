from multiprocessing import Queue
from queue import Empty
import cv2
from flask import Flask, render_template, request, redirect
import base64

from collections import deque
from dataclasses import dataclass
from multiprocessing import Queue, Event
from typing import List, Optional
from matplotlib import pyplot as plt
import numpy as np
import random
from numpy.typing import NDArray
import torch
from drlhp.reward_predictor import PreferenceDatabase

from threading import Lock, Thread

app = Flask(__name__)


@dataclass
class ObservationForPreference:
    rgb: NDArray[np.uint8]
    observation: NDArray[np.float32]
    action: int
    environment_reward: float


@dataclass
class TrajectoryForPreference:
    obs_actions1: list[tuple[torch.Tensor, int]]
    obs_actions2: list[tuple[torch.Tensor, int]]
    env_rewards1: list[float]
    env_rewards2: list[float]


@dataclass
class ClipAndPref:
    clip1: list[ObservationForPreference]
    clip2: list[ObservationForPreference]
    preference: float


class SegmentDatabase:
    def __init__(self, size: int, fps: int = 30, segment_length_s: float = 1.5, lock: Optional[Lock] = None):
        self.queue = deque(maxlen=size)
        self.fps: int = fps
        self.segment_lenght_s: float = segment_length_s
        self.num_frames_per_segment: int = int(segment_length_s * fps)
        self.preference_database = PreferenceDatabase()
        self.lock = lock

    def is_full(self) -> bool:
        return len(self.queue) == self.queue.maxlen

    def drop_oldest(self, num_to_drop: int = 1):
        if self.lock is not None:
            with self.lock:
                for _ in range(num_to_drop):
                    if len(self.queue) > 0:
                        self.queue.popleft()

    def add_trajectory(self, trajectory: List[ObservationForPreference]):
        # trajectory is a list of rgb arrays
        # from trajectory need to extract a series of frames

        len_clip = min(self.num_frames_per_segment, len(trajectory))
        if len(trajectory) > self.num_frames_per_segment:
            start_point = random.randrange(0, len(trajectory) - self.num_frames_per_segment)
        else:
            start_point = 0

        clip = trajectory[start_point : start_point + len_clip]

        if self.lock is not None:
            with self.lock:
                self.queue.append(clip)
        else:
            self.queue.append(clip)

    def get_random_clip(self) -> List[ObservationForPreference]:
        return random.choice(self.queue)

    def get_trajectories_for_preference(self) -> tuple[list[ObservationForPreference], list[ObservationForPreference]]:
        return random.sample(self.queue, 2)


def numpy_to_base64(img_array: NDArray[np.uint8]):
    # Convert the NumPy array to JPEG using OpenCV
    assert img_array.sum() > 255, "Image should have something in it"

    _, img_encoded = cv2.imencode(".jpg", img_array)

    # Encode the JPEG data in base64 and return
    return base64.b64encode(img_encoded).decode("utf-8")


class FlaskWrapper:
    def __init__(self, segment_database: SegmentDatabase, preference_queue: Queue):
        self.app = Flask(__name__)
        self.clip_and_pref = None

        # Assuming you have configurations you want to use
        # self.app.config["some_key"] = "some_value"
        self.app.config["segment_database"] = segment_database
        self.app.config["preference_queue"] = preference_queue

        self.configure_routes()
        self.add_no_cache_headers()

    def add_no_cache_headers(self):
        @self.app.after_request
        def add_no_cache_headers(response):
            response.headers[
                "Cache-Control"
            ] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "-1"
            return response

    def configure_routes(self):
        @self.app.route("/")
        def index():
            segment_database: SegmentDatabase = self.app.config["segment_database"]
            print("starting index")

            clip1, clip2 = segment_database.get_trajectories_for_preference()
            print("getting new trajectories")
            self.clip_and_pref = ClipAndPref(clip1, clip2, 0.5)

            clip1_data = [numpy_to_base64(frame) for frame in [obs.rgb for obs in clip1]]
            clip2_data = [numpy_to_base64(frame) for frame in [obs.rgb for obs in clip2]]

            return render_template("index.html", clip1_data=clip1_data, clip2_data=clip2_data, frameRate=100)

        @self.app.route("/register_preference", methods=["POST"])
        def register_preference():
            preference_queue: Queue = self.app.config["preference_queue"]

            preference = request.json["preference"]
            if preference == "clip1":
                preference = 0
            elif preference == "clip2":
                preference = 1
            else:
                preference = 0.5

            self.clip_and_pref.preference = preference
            print("preference_queue.put")
            preference_queue.put(self.clip_and_pref)
            self.clip_and_pref = None
            print(f"Preference received: {preference}")

            return redirect("/")  # Redirect back to the main route to fetch new clips

    def run(self, **kwargs):
        self.app.run(**kwargs)


def play_frames_as_video(frames):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Play video from frames
    for frame in frames:
        if "im" in locals():
            im.set_data(frame)
        else:
            im = ax.imshow(frame)

        plt.pause(1 / 30)  # Assuming 30fps, adjust based on desired frame rate

    plt.close()


def preference_interface_loop(input_queue: Queue, preference_queue: Queue, should_exit: Event):  # type: ignore
    # Start the Flask app and pass the input_queue
    lock = Lock()
    segment_database = SegmentDatabase(size=100, lock=lock)

    # # Start the Flask app in a separate thread
    flask_wrapper = FlaskWrapper(segment_database, preference_queue)

    flask_thread = Thread(target=flask_wrapper.run)
    flask_thread.start()

    while not should_exit.is_set():
        # check if there is a new observation
        try:
            observation = input_queue.get(timeout=1)
            print("Preference process received observation")
            if segment_database.is_full():
                segment_database.drop_oldest()
            segment_database.add_trajectory(observation)

            # clip1, clip2 = segment_database.get_trajectories_for_preference()
            # play_frames_as_video([obs.rgb for obs in clip1])

        except Empty:
            pass
