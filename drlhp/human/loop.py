"""
This module recieves clips from the environment loop
It runs the Flask app and then passes thoose clips to the app and then sends the rated ones to the preference model loop
"""

import queue
import random
from collections import deque
from multiprocessing.synchronize import Event
from threading import Lock, Thread

from slist import Slist

from drlhp.comms import Observation, PairedObservations, TypedQueue
from drlhp.human.web_interface.app import FlaskWrapper


class SegmentDatabase:
    def __init__(self, size: int, fps: int = 30, segment_length_s: float = 2):
        self.queue: deque[Slist[Observation]] = deque(maxlen=size)
        self.fps: int = fps
        self.segment_lenght_s: float = segment_length_s
        self.num_frames_per_segment: int = int(segment_length_s * fps)
        self.out_queue: deque[PairedObservations] = deque(maxlen=10)
        self.lock = Lock()

    def is_full(self) -> bool:
        return len(self.queue) == self.queue.maxlen

    def drop_oldest(self, num_to_drop: int = 1):
        for _ in range(num_to_drop):
            if len(self.queue) > 0:
                self.queue.popleft()

    def add(self, trajectory: Slist[Observation]):
        # trajectory is a list of rgb arrays
        # from trajectory need to extract a series of frames

        len_clip = min(self.num_frames_per_segment, len(trajectory))
        if len(trajectory) > self.num_frames_per_segment:
            start_point = random.randrange(0, len(trajectory) - self.num_frames_per_segment)
        else:
            start_point = 0

        clip = trajectory[start_point : start_point + len_clip]

        with self.lock:
            self.queue.append(clip)

    def get(self) -> PairedObservations | None:
        if len(self.queue) < 2:
            return None
        with self.lock:
            sampled: list[Slist[Observation]] = random.sample(self.queue, 2)

        print(f"SegmentDatabase: sampled two clips, length: {len(self.queue)}")
        return PairedObservations(obs_1=sampled[0], obs_2=sampled[1])


def preference_interface_loop(
    input_queue: TypedQueue[Slist[Observation]],
    output_queue: TypedQueue[PairedObservations],
    should_exit: Event,
):  # type: ignore
    """
    Waits on the input queue, queries a human for the preferences, and then puts the result on the output queue
    """

    # Start the Flask app and pass the input_queue
    segment_database = SegmentDatabase(size=10)

    rated_queue: deque[PairedObservations] = deque()

    # # Start the Flask app in a separate thread
    flask_wrapper = FlaskWrapper(get_from_source=segment_database.get, add_to_sink=rated_queue.append)
    flask_thread = Thread(target=flask_wrapper.run)
    flask_thread.start()

    while not should_exit.is_set():
        try:
            observation = input_queue.get(timeout=1)
        except queue.Empty:
            continue
        print("Preference process received observation")
        segment_database.add(observation)

        if rated_queue:
            output_queue.put(rated_queue.popleft())
