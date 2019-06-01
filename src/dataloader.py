from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import logging

logging.getLogger("PIL").setLevel(logging.WARNING)


class LoaderHelper(Dataset):

    def __init__(self, image_paths, targets, **kwargs):

        self._image_paths = image_paths
        self._targets = targets
        self._transform = kwargs.get("transform", None)

    def __getitem__(self, idx):
        data = Image.open(self._image_paths[idx])
        target = int(self._targets[idx])
        if self._transform:
            data = self._transform(data)
        return data, target

    def __len__(self):

        return len(self._image_paths)

    @property
    def targets(self):
        return self._targets

    @property
    def data(self):
        return self._image_paths


class DirDataset(LoaderHelper):

    def __init__(self, path, **kwargs):

        image_paths, targets = self._load_paths(path)
        super().__init__(image_paths, targets, **kwargs)

    def _load_paths(self, path):
        image_paths = []
        targets = []
        assert os.path.isdir(path)
        for directory in glob(path + "/*"):
            assert os.path.isdir(directory)
            dir_images = self._scan_dir(directory)
            image_paths.extend(dir_images)
            target = os.path.basename(directory)
            targets.extend([target] * len(dir_images))

        assert len(image_paths) == len(targets)
        return image_paths, targets

    def _scan_dir(self, directory):
        image_types = ('*.jpeg', '*.jpg', '*.png')
        image_list = []
        for image_type in image_types:
            image_list.extend(glob(directory + "/" + image_type))

        return image_list


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to images")
    args = parser.parse_args()

    loader = DirDataset(args.path)
