import pickle
from glob import glob
from torch.utils.data import Dataset


class PanAfDataset(Dataset):
    def __init__(self, data_dir, ann_dir, sequence_len):
        super(PanAfDataset, self).__init__()

        self.videos = glob.glob(f"{data_dir}/**/*.mp4", recursive=True)
        self.anns = glob.glob(f"{ann_dir}/**/*.pkl", recursive=True)
        assert(len(self.videos) == len(self.anns))

        self.sequence_len = sequence_len
    
    def count_apes(self, data):
        ids = []
        for frame in data['annotations']:
            for detection in frame['detections']:
                ids.append(detection['ape_id'])
        return(max(ids))

    def initialise_dataset(self):
        for video in self.videos:
            with open(video, 'rb') as handle:
                data = pickle.load(handle)
            
            no_of_apes = self.count_apes(data)
