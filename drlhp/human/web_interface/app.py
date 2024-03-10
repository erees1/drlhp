"""
This module contains the implementation of the Human Interface, which is a flask app
that shows two clips from a database and threading
once a preference is recieved, stick it back into the database
"""

from collections import deque
from collections.abc import Callable
from time import time

import numpy as np
from flask import Flask, redirect, render_template, request
from slist import Slist

from drlhp.comms import Observation, PairedObservations
from drlhp.human.web_interface.img_utils import numpy_to_base64

app = Flask(__name__)


class FlaskWrapper:
    def __init__(
        self,
        get_from_source: Callable[[], PairedObservations | None],
        add_to_sink: Callable[[PairedObservations], None],
        viewer_fps: int = 30,
    ):
        self.app = Flask(__name__)
        self.paired_observations: PairedObservations | None = None

        # Assuming you have configurations you want to use
        # self.app.config["some_key"] = "some_value"
        self.get_from_source = get_from_source
        self.add_to_sink = add_to_sink
        self.viewer_fps = viewer_fps

        self.configure_routes()

    def configure_routes(self):
        @self.app.route("/")
        def index():
            print("getting new trajectories")
            try:
                self.paired_observations = self.get_from_source()
            except IndexError:
                self.paired_observations = None

            if self.paired_observations is None:
                print("to_rate_queue is empty")
                return "No clips to rate"

            clip1_data = self.paired_observations.obs_1.map(lambda x: numpy_to_base64(x.rgb))
            clip2_data = self.paired_observations.obs_2.map(lambda x: numpy_to_base64(x.rgb))

            return render_template(
                "index.html", clip1_data=clip1_data, clip2_data=clip2_data, frameRate=self.viewer_fps
            )

        @self.app.route("/register_preference", methods=["POST"])
        def register_preference():
            # Retrieving which button was clicked
            button_clicked = request.form["button"]
            print(f"Button clicked: {button_clicked}")

            if button_clicked == "clip1":
                preference = 0
            elif button_clicked == "clip2":
                preference = 1
            else:
                preference = 0.5

            assert self.paired_observations is not None, "No clip to rate"
            self.paired_observations.preference = preference
            print("preference_queue.put")
            self.add_to_sink(self.paired_observations)
            self.paired_observations = None
            print(f"Preference received: {preference}")

            # Redirect back to the main route to fetch new clips
            # add time to avoid caching
            return redirect(f"/?time={time()}")

        return index, register_preference

    def run(self):
        self.app.run()


if __name__ == "__main__":
    # dummy data
    clip1_rgb = Slist([np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(128)])
    clip2_rgb = Slist([np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(128)])

    dummy_obs_1 = clip1_rgb.map(
        lambda x: Observation(rgb=x, observation=x.astype(np.float32), action=0, environment_reward=0)
    )
    dummy_obs_2 = clip1_rgb.map(
        lambda x: Observation(rgb=x, observation=x.astype(np.float32), action=0, environment_reward=0)
    )
    to_rate_queue = deque([PairedObservations(obs_1=dummy_obs_1, obs_2=dummy_obs_2)])
    rated_queue = deque()

    FlaskWrapper(get_from_source=to_rate_queue.popleft, add_to_sink=rated_queue.append).run()

    print("rated items:", rated_queue)
