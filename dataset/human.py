import mmcv
from tqdm import tqdm
from dataset.base import PanAfDataset
from typing import Callable, Optional


class SupervisedPanAf(PanAfDataset):
    """
    A Pan-African Pytorch Dataset class for supervised training.

    Args:

     Paths:
         data_dir: Path to video files (i.e., 'data/train').
         ann_dir: Path to annotation files (i.e., 'annotations/train')

     Sample building:

         sequence_len: Number of frames in each sample. The output tensor
         will have shape (B x C x T x W x H) where B = batch_size, C = channels,
         T = sequence_len, W = width and H = height.

         sample_itvl: Number of frames between each sample frame i.e., if
         sample_itvl = 1 consecutive frames are sampled, if sample_itvl = 2
         every other frame is sampled.

         *Note if sequence_len = 5 and sample_itvl = 2, the output tensor will
         be of shape (B x C x 5 x H x W) which is sampled from a tensor of
         shape (B x C x 10 x H x W).

         stride: Number of frames between samples. By default, this is
         sequence_len x sample_itvl. This means samples are built consecutively
         and without overlap. If the stride is manually set to a lower value
         samples will be generated with overlapping frames i.e., samples built
         with sequence_len = 20 and stride = 10 will have a 10-frame overlap.

         behaviour_threshold: Number of frames an ape must be both present and
         displaying a specific behaviour for a sample to be valid.

     Transform:
         transform: List of transforms to be applied.
    """

    def __init__(
        self,
        data_dir: str = ".",
        ann_dir: str = ".",
        dense_dir: str = ".",
        sequence_len: int = 5,
        sample_itvl: int = 1,
        stride: int = None,
        type: str = "r",
        transform: Optional[Callable] = None,
        behaviour_threshold: int = 72,
        split: str = None,
    ):
        self.behaviour_threshold = behaviour_threshold
        self.split = split

        self.classes = {
            "camera_interaction": 0,
            "climbing_down": 1,
            "climbing_up": 2,
            "hanging": 3,
            "running": 4,
            "sitting": 5,
            "sitting_on_back": 6,
            "standing": 7,
            "walking": 8,
        }

        super().__init__(
            data_dir,
            ann_dir,
            dense_dir,
            sequence_len,
            sample_itvl,
            stride,
            type,
            transform,
        )

        self.samples_by_class()

    def get_behaviour_index(self, behaviour):
        return self.classes[behaviour]

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

    def get_valid_frames(
        self, ann, current_ape, current_behaviour, frame_no, no_of_frames
    ):
        valid_frames = 1
        for look_ahead_frame_no in range(frame_no + 1, no_of_frames + 1):
            ape = self.check_ape_exists(ann, look_ahead_frame_no, current_ape)

            if (ape) and (
                self.get_ape_behaviour(ann, current_ape, look_ahead_frame_no)
                == current_behaviour
            ):
                if "d" not in self.type:
                    valid_frames += 1
                else:
                    dense = self.check_dense_exists(
                        ann, look_ahead_frame_no, current_ape
                    )
                    if dense:
                        valid_frames += 1
            else:
                return valid_frames

        return valid_frames

    def samples_by_class(self):
        self.samples_by_class = {}
        for sample in self.samples:
            behaviour = sample['behaviour']
            if(behaviour not in self.samples_by_class.keys()):
                self.samples_by_class[behaviour] = 1
            else:
                self.samples_by_class[behaviour] += 1
        return

    def print_samples_by_class(self):
        print(self.samples_by_class)

    def initialise_dataset(self):
        for data in tqdm(self.data):

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

                    valid_frames = self.get_valid_frames(
                        ann, current_ape, current_behaviour, frame_no, no_of_frames
                    )

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
        behaviour_idx = self.get_behaviour_index(behaviour)
        video = self.get_video(name)
        spatial_sample = self.build_spatial_sample(video, name, ape_id, frame_idx)
        return spatial_sample, behaviour_idx
