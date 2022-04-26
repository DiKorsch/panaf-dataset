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
    ):
        self.behaviour_threshold = behaviour_threshold
        super().__init__(
            data_dir, ann_dir, sequence_len, sample_itvl, stride, transform
        )

    def get_ape_behaviour(self, ann, current_ape, frame_no):
        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape:
                        return d["behaviour"]
        return None

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

            # Check no of frames match
            assert len(video) == len(ann["annotations"])

            no_of_apes = self.count_apes(ann)
            # TODO: check all apes index from 0...

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

                    sufficient_apes = self.check_sufficient_apes(
                        ann, current_ape, frame_no
                    )

                    if not sufficient_apes:
                        frame_no += 1  # self.sequence_len
                        continue

                    behaviour = self.get_ape_behaviour(ann, current_ape, frame_no)

                    behaviour_threshold = self.check_behaviour_threshold(
                        ann, current_ape, frame_no, behaviour
                    )

                    if not behaviour_threshold:
                        frame_no += 1
                        continue

                    if (len(video) - frame_no) >= self.sequence_len:
                        self.samples.append(
                            {
                                "video": name,
                                "ape_id": current_ape,
                                "behaviour": behaviour,
                                "start_frame": frame_no,
                            }
                        )
                    frame_no += self.stride
        return

    def __getitem__(self, index):
        sample = self.samples[index]
        ape_id = sample["ape_id"]
        frame_idx = sample["start_frame"]
        name = sample["video"]
        behaviour = sample["behaviour"]
        video = self.get_video(name)
        spatial_sample = self.build_spatial_sample(video, name, ape_id, frame_idx)
        return spatial_sample, sample
