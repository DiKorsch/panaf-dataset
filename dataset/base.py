import json
import mmcv
import torch
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset

# TODO: Want this to be our base class (consider base for human and machine...)
# TODO: Optimise fetching of bboxes (numba, cython?)
# TODO: Behaviour threshold
# TODO: Load dense annotations


class PanAfDataset(Dataset):
    def __init__(self, data_dir, ann_dir, sequence_len, sample_itvl, transform):
        super(PanAfDataset, self).__init__()

        self.data_path = data_dir
        self.ann_path = ann_dir
        self.data = glob(f"{data_dir}/**/*.mp4", recursive=True)
        self.anns = glob(f"{ann_dir}/**/*.json", recursive=True)
        # assert len(self.data) == len(self.anns), f"{len(self.data)}, {len(self.anns)}"

        # Number of frames in sequence
        self.sequence_len = sequence_len

        # Number of in-between frames
        self.sample_itvl = sample_itvl

        # Frames required to build samples
        self.total_seq_len = sequence_len * sample_itvl

        self.transform = transform

        self.samples = []
        self.initialise_dataset()

    def get_videoname(self, path):
        return path.split("/")[-1].split(".")[0]

    def count_apes(self, ann):
        # TODO: check all videos index from ape 0, 1, 2...
        ids = []
        for frame in ann["annotations"]:
            for detection in frame["detections"]:
                ids.append(detection["ape_id"])

        if not ids:
            return False

        return max(ids)

    def check_ape_exists(self, ann, frame_no, current_ape):
        ape = False

        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape:
                        ape = True
        return ape

    def check_sufficient_apes(self, ann, current_ape, frame_no):
        for look_ahead_frame_no in range(frame_no, frame_no + self.total_seq_len):
            ape = self.check_ape_exists(ann, look_ahead_frame_no, current_ape)
            if not ape:
                return False
        return True

    def get_ape_behaviour(self, ann, current_ape, frame_no):
        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape:
                        return d["behaviour"]

    def check_behaviour_threshold(self, ann, current_ape, frame_no):
        try:
            behaviour = self.get_ape_behaviour(ann, current_ape, frame_no)
        except ValueError:
            print("No behaviour found!")

        for look_ahead_frame_no in range(frame_no, frame_no + self.sequence_len):
            future_behaviour = self.get_ape_behaviour(ann, current_ape, frame_no)
            if future_behaviour != behaviour:
                return False
        return True

    def load_annotation(self, filename):
        with open(f"{self.ann_path}/{filename}.json", "rb") as handle:
            ann = json.load(handle)
        return ann

    def print_samples(self):
        print(self.samples)

    def __len__(self):
        return len(self.samples)

    def initialise_dataset(self):
        for data in tqdm(self.data[:100], desc="Initialising samples", leave=False):

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
                    ) < self.total_seq_len - 1:  # TODO: check equality symbol is correct
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

                    if (len(video) - frame_no) >= self.sequence_len:
                        self.samples.append(
                            {
                                "video": name,
                                "ape_id": current_ape,
                                "start_frame": frame_no,
                            }
                        )
                    frame_no += self.sequence_len
        return

    def get_ape_coords(self, video, ape_id, frame_idx):
        with open(f"{self.ann_path}/{video}.json", "rb") as handle:
            ann = json.load(handle)
        try:
            for a in ann["annotations"]:
                if a["frame_id"] == frame_idx:
                    for d in a["detections"]:
                        if d["ape_id"] == ape_id:
                            return d["bbox"]
        except ValueError:
            print(f"{video} {frame_idx}: couldnt find bbox for ape {ape_id}.")

    def __getitem__(self, index):
        sample = self.samples[index]
        ape_id = sample["ape_id"]
        frame_idx = sample["start_frame"]
        name = sample["video"]
        for video_path in self.data:
            if self.get_videoname(video_path) == name:
                video = mmcv.VideoReader(video_path)
                break

        spatial_sample = []

        for i in range(0, self.total_seq_len, self.sample_itvl):
            spatial_img = video[frame_idx + i - 1]
            coords = list(map(int, self.get_ape_coords(name, ape_id, frame_idx + i)))
            cropped_img = spatial_img[coords[1] : coords[3], coords[0] : coords[2]]
            spatial_data = self.transform(cropped_img)
            spatial_sample.append(spatial_data.squeeze_(0))
        spatial_sample = torch.stack(spatial_sample, dim=0)
        spatial_sample = spatial_sample.permute(0, 1, 2, 3)

        assert len(spatial_sample) == self.sequence_len

        return spatial_sample
