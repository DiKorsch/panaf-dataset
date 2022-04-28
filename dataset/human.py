import mmcv
from tqdm import tqdm
from dataset.base import PanAfDataset
from typing import Callable, Optional


class SupervisedPanAf(PanAfDataset):
    def __init__(
        self,
        data_dir: str = ".",
        ann_dir: str = ".",
        sequence_len: int = 5,
        sample_itvl: int = 1,
        stride: int = None,
        transform: Optional[Callable] = None,
        behaviour_threshold: int = 72,
        split: str = None,
    ):
        self.behaviour_threshold = behaviour_threshold
        self.split = split

        super().__init__(
            data_dir, ann_dir, sequence_len, sample_itvl, stride, transform
        )

    def count_videos(self):
        return len(self.samples_by_video)

    def get_ape_behaviour(self, ann, current_ape, frame_no):
        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape:
                        return d["behaviour"]
        return None

    def set_behaviour_threshold(self, value):
        self.behaviour_threshold = value

    def check_behaviour_threshold(self, ann, current_ape, frame_no, target_behaviour):
        for look_ahead_frame_no in range(frame_no, frame_no + self.behaviour_threshold):
            future_behaviour = self.get_ape_behaviour(
                ann, current_ape, look_ahead_frame_no
            )
            if future_behaviour != target_behaviour:
                return False
        return True

    def initialise_dataset(self):
        for data in tqdm(self.data, desc="Initialising samples", leave=False):

            name = self.get_videoname(data)
            video = mmcv.VideoReader(data)
            ann = self.load_annotation(name)
            no_of_frames = len(video)
            # Check no of frames match
            assert len(video) == len(ann["annotations"])

            no_of_apes = self.count_apes(ann)

            for current_ape in range(0, no_of_apes + 1):
                frame_no = 1

                while frame_no <= len(video):
                    if (
                        len(video) - frame_no
                    ) < self.total_seq_len:  # TODO: check equality symbol is correct
                        break

                    ape = self.check_ape_exists(ann, frame_no, current_ape)

                    if not ape:
                        frame_no += 1
                        continue

                    current_behaviour = self.get_ape_behaviour(
                        ann, current_ape, frame_no
                    )
                    valid_frames = 1

                    for look_ahead_frame_no in range(frame_no + 1, no_of_frames + 1):
                        ape = self.check_ape_exists(
                            ann, look_ahead_frame_no, current_ape
                        )

                        if (ape) and (
                            self.get_ape_behaviour(
                                ann, current_ape, look_ahead_frame_no
                            )
                            == current_behaviour
                        ):
                            valid_frames += 1
                        else:
                            break

                    if valid_frames < self.behaviour_threshold:
                        frame_no += valid_frames
                        continue

                    last_valid_frame = frame_no + valid_frames

                    for valid_frame_no in range(
                        frame_no, last_valid_frame, self.stride
                    ):
                        if (valid_frame_no + self.stride) >= last_valid_frame:
                            correct_activity = False

                            for temporal_frame in range(
                                valid_frame_no, self.total_seq_len
                            ):
                                ape = self.check_ape_exists(
                                    ann, temporal_frame, current_ape
                                )
                                ape_activity = self.get_ape_behaviour(
                                    ann, temporal_frame, current_ape
                                )
                                if (
                                    (not ape)
                                    or (ape_activity != current_behaviour)
                                    or (temporal_frame > no_of_frames)
                                ):
                                    correct_activity = False
                                    break
                            if not correct_activity:
                                break

                        if (no_of_frames - valid_frame_no) >= self.total_seq_len:
                            self.samples.append(
                                {
                                    "video": name,
                                    "ape_id": current_ape,
                                    "behaviour": current_behaviour,
                                    "start_frame": valid_frame_no,
                                }
                            )

                    frame_no = last_valid_frame

    def __getitem__(self, index):
        sample = self.samples[index]
        ape_id = sample["ape_id"]
        frame_idx = sample["start_frame"]
        name = sample["video"]
        behaviour = sample["behaviour"]
        video = self.get_video(name)
        spatial_sample = self.build_spatial_sample(video, name, ape_id, frame_idx)
        return spatial_sample, sample
