import os
import json
import mmcv
import torch
import pickle
import torchvision
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from typing import Callable, Optional

from torchvision.transforms.functional import resize

# TODO: Want this to be our base class (consider base for human and machine...)
# TODO: Optimise fetching of bboxes (numba, cython?)
# TODO: Load dense annotations


class PanAfDataset(Dataset):
    """
    Base class for Pan-African Pytorch Dataset

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

     Transform:
         transform: List of transforms to be applied.
    """

    def __init__(
        self,
        data_dir: str = None,
        ann_dir: str = None,
        dense_dir: str = None,
        flow_dir: str = None,
        sequence_len: int = None,
        sample_itvl: int = None,
        stride: int = None,
        type: str = "",
        behaviour_threshold: int = None,
        split: str = None,
        spatial_transform: Optional[Callable] = None,
        temporal_transform: Optional[Callable] = None,
        which_classes: Optional[str] = None,
    ):
        # TODO: include behaviour_thresh in init
        super(PanAfDataset, self).__init__()

        self.data_path = data_dir
        self.ann_path = ann_dir
        self.dense_dir = dense_dir
        self.flow_dir = flow_dir
        self.data = glob(f"{data_dir}/**/*.mp4", recursive=True)
        self.anns = glob(f"{ann_dir}/**/*.json", recursive=True)
        self.dense_data = glob(f"{dense_dir}/**/*.pkl", recursive=True)

        # assert len(self.data) == len(self.anns), f"{len(self.data)}, {len(self.anns)}"

        # Number of frames in sequence
        self.sequence_len = sequence_len

        # Number of in-between frames
        self.sample_itvl = sample_itvl

        # Frames required to build samples
        self.total_seq_len = sequence_len * sample_itvl

        # Frames between samples
        if stride is None:
            # This leaves no overlap between samples
            self.stride = self.total_seq_len
        else:
            self.stride = stride

        self.type = [c for c in type]

        self.behaviour_threshold = behaviour_threshold

        self.split = split
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.which_classes = which_classes

        self.samples = {}
        self.labels = 0
        self.initialise_dataset()

        self.samples_by_video = {}

    def get_videoname(self, path):
        return path.split("/")[-1].split(".")[0]

    def get_dense_ann_name(self, path):
        return path.split("/")[-1].split(".")[0].split("_")[0]

    def verify_ape_ids(self, no_of_apes, ids):
        for i in range(0, no_of_apes + 1):
            if i not in ids:
                return False
        return True

    def count_apes(self, ann):
        ids = []
        for frame in ann["annotations"]:
            for detection in frame["detections"]:
                ids.append(detection["ape_id"])

        if not ids:
            return None

        assert self.verify_ape_ids(
            max(ids), list(set(ids))
        ), f"{ann['video'], max(ids), list(set(ids))}"

        return max(ids)

    def check_ape_exists(self, ann, frame_no, current_ape):
        ape = False
        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape:
                        ape = True
        return ape

    def check_dense_exists(self, ann, frame_no, current_ape):
        dense = False
        for a in ann["annotations"]:
            if a["frame_id"] == frame_no:
                for d in a["detections"]:
                    if d["ape_id"] == current_ape and "dense" in d.keys():
                        dense = True
        return dense

    def check_sufficient_apes(self, ann, current_ape, frame_no):
        for look_ahead_frame_no in range(frame_no, frame_no + self.total_seq_len):
            ape = self.check_ape_exists(ann, look_ahead_frame_no, current_ape)
            if not ape:
                return False
        return True

    def load_annotation(self, filename: str = None):
        if "d" not in self.type:
            with open(f"{self.ann_path}/{filename}.json", "rb") as handle:
                ann = json.load(handle)
        else:
            with open(f"{self.dense_dir}/{filename}_dense.pkl", "rb") as handle:
                ann = pickle.load(handle)
        return ann

    def print_samples(self):
        print(self.samples)

    def __len__(self):
        samples = 0
        for video in self.samples.keys():
            samples += len(self.samples[video])
        return samples

    def initialise_dataset(self):
        for data in self.data:

            name = self.get_videoname(data)
            video = mmcv.VideoReader(data)
            ann = self.load_annotation(name)

            # Check no of frames match
            assert len(video) == len(ann["annotations"])

            no_of_apes = self.count_apes(ann)
            if no_of_apes is None:
                break

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

                    if (len(video) - frame_no) >= self.sequence_len:

                        if name not in self.samples.keys():
                            self.samples[name] = []

                        self.labels += 1

                        self.samples[name].append(
                            {
                                "video": name,
                                "ape_id": current_ape,
                                "start_frame": frame_no,
                            }
                        )
                    frame_no += self.stride
        return

    # Get the ith sample from the dataset
    def find_sample(self, index):
        current_index = 0

        for key in self.samples.keys():
            for i, value in enumerate(self.samples[key]):
                if current_index == index:
                    return (
                        self.samples[key][i]["video"],
                        self.samples[key][i]["ape_id"],
                        self.samples[key][i]["start_frame"],
                    )
                current_index += 1

    def get_ape_coords(self, video, ape_id, frame_idx):
        bbox = None

        with open(f"{self.ann_path}/{video}.json", "rb") as handle:
            ann = json.load(handle)

        for a in ann["annotations"]:
            if a["frame_id"] == frame_idx:
                for d in a["detections"]:
                    if d["ape_id"] == ape_id:
                        bbox = d["bbox"]
        return bbox

    def build_temporal_sample(self, name, ape_id, frame_idx):
        temporal_sample = []
        for i in range(0, self.total_seq_len, self.sample_itvl):
            h_flow = Image.open(
                f"{self.flow_dir}/horizontal_flow/{name}/{name}_frame_{frame_idx + i}.jpg"
            )
            v_flow = Image.open(
                f"{self.flow_dir}/vertical_flow/{name}/{name}_frame_{frame_idx + i}.jpg"
            )
            coords = list(map(int, self.get_ape_coords(name, ape_id, frame_idx + i)))
            h_flow_cropped = self.temporal_transform(h_flow.crop(coords))
            v_flow_cropped = self.temporal_transform(v_flow.crop(coords))
            hv_flow = torch.stack(
                (h_flow_cropped.squeeze_(0), v_flow_cropped.squeeze_(0)), dim=0
            )
            temporal_sample.append(hv_flow)
        temporal_sample = torch.stack(temporal_sample, dim=0)
        assert (
            len(temporal_sample) == self.sequence_len
        ), f"video: {name}, ape_id: {ape_id}, frame_idx: {frame_idx}"
        return temporal_sample

    def build_spatial_sample(self, video, name, ape_id, frame_idx):
        spatial_sample = []
        for i in range(0, self.total_seq_len, self.sample_itvl):
            frame = video[frame_idx + i - 1].numpy()
            spatial_img = Image.fromarray(frame)
            coords = list(map(int, self.get_ape_coords(name, ape_id, frame_idx + i)))
            cropped_img = spatial_img.crop(coords)
            try:
                spatial_data = self.spatial_transform(cropped_img)
            except:
                raise ValueError(
                    f"Name: {name}, Ape: {ape_id}, Frame: {frame_idx + i}, Box: {coords}"
                )
            spatial_sample.append(spatial_data.squeeze_(0))
        spatial_sample = torch.stack(spatial_sample, dim=0)
        spatial_sample = spatial_sample.permute(0, 1, 2, 3)

        # Check frames in sample match sequence length
        assert (
            len(spatial_sample) == self.sequence_len
        ), f"video: {name}, ape_id: {ape_id}, frame_idx: {frame_idx}"

        return spatial_sample

    def check_segmentation(self, det):
        seg = False
        if "labels" in det.keys():
            seg = True
        return seg

    def get_segmentation(self, det):
        return det["labels"][None, :, :]

    def get_uv(self, det):
        return det["uv"]

    def build_iuv(self, i, uv):
        iuv = torch.cat(i[None].type(torch.float32), uv * 255.0).type(torch.uint8)
        return iuv

    def resize_stack_seq(self, sequence, size):
        sequence = [resize(x, size=size).squeeze(dim=0) for x in sequence]
        return torch.stack(sequence, dim=0)

    def get_frame(self, ann, frame_idx):
        frame = None
        for i in range(len(ann["annotations"])):
            if ann["annotations"][i]["frame_id"] == frame_idx:
                frame = ann["annotations"][i]["frame_id"]
        return frame

    def build_dense_sample(self, ann, name, ape_id, frame_idx):

        dense_sample = list()

        assert ann["video"] == name
        for i in range(len(ann["annotations"])):
            if ann["annotations"][i]["frame_id"] == frame_idx:
                for j in range(0, self.total_seq_len, self.sample_itvl):
                    for det in ann["annotations"][i + j]["detections"]:
                        if (det["ape_id"] == ape_id) and (self.check_segmentation(det)):
                            seg = self.get_segmentation(det)
                            uv = self.get_uv(det)
                            iuv = torch.cat((seg, uv), dim=0)[None]
                            iuv = resize(iuv, size=(244, 244)).squeeze(dim=0)
                            dense_sample.append(iuv)
                            if len(dense_sample) == self.sequence_len:
                                dense_sample = torch.stack(dense_sample, dim=0)
                                # Check frames in sample match sequence length
                                assert (
                                    len(dense_sample) == self.sequence_len
                                ), f"{len(dense_sample)} != {self.sequence_len}"
                                return dense_sample

    def build_sample(self, name, ape_id, frame_idx):
        sample = dict()
        if "r" in self.type:
            video = self.get_video(name)
            sample["spatial_sample"] = self.build_spatial_sample(
                video, name, ape_id, frame_idx
            )
        if "d" in self.type:
            dense_annotation = self.get_dense_annotation(name)
            sample["dense_sample"] = self.build_dense_sample(
                dense_annotation, name, ape_id, frame_idx
            )
        if "f" in self.type:
            sample["flow_sample"] = self.build_temporal_sample(name, ape_id, frame_idx)
        return sample

    def get_video(self, name):
        video = None
        for video_path in self.data:
            if self.get_videoname(video_path) == name:
                if os.path.isfile(video_path):
                    video = torchvision.io.read_video(
                        filename=video_path, pts_unit="sec"
                    )
                else:
                    print(f"Path error: {video_path}")
        if video is not None:
            video = video[0]
        return video

    def get_dense_annotation(self, name):
        dense_annotation = None
        for path in self.dense_data:
            filename = self.get_dense_ann_name(path)
            if filename == name:
                dense_annotation = self.load_annotation(name)
        return dense_annotation

    def __getitem__(self, index):
        name, ape_id, frame_idx = self.find_sample(index)
        sample = self.build_sample(name, ape_id, frame_idx)
        return sample
