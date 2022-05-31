import torch as th 

from os import path 
from torch.utils.data import Dataset
from libraries.strategies import pull_images, read_image, prepare_image, deserialize, cv2th

class DatasetForFeaturesExtraction(Dataset):
    def __init__(self, path2images, file_extension='*.jpg'):
        self.image_paths = pull_images(path2images, exts=file_extension)
        self.image_names = [ path.split(path_)[1] for path_ in self.image_paths ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path2img = self.image_paths[index]
        cv_image = read_image(path2img)
        th_image = cv2th(cv_image)
        return prepare_image(th_image)

class DatasetForTraining(Dataset):
    def __init__(self, path2tokenids, path2features):
        self.tokenids = deserialize(path2tokenids)
        self.features = deserialize(path2features)
    def __len__(self):
        return len(self.tokenids)
    
    def __getitem__(self, idx):
        file_name, ids = self.tokenids[idx]
        vec = self.features[file_name]
        vec = th.tensor(vec).float()
        ids = th.tensor(ids).long()
        return vec, ids