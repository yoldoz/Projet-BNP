from pathlib import Path
import shutil
from torch.utils.data import Dataset

import os
import glob
import json 
import random
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython.display import display
import matplotlib
from matplotlib import pyplot, patches

from utils_sroie import read_bbox_and_words, read_entities, assign_line_label, assign_labels, split_line, normalize


class SROIE(Dataset):
    def __init__(
        self,
        run,
        data_path,
        config,
        transform,
    ):
        self.data_path = Path.cwd() / data_path
        if self.data_path.exists() and self.data_path.is_dir():
            shutil.rmtree(self.data_path)
        self.data_path.mkdir(exist_ok=True)
        artifact = run.use_artifact(
            f"{run.entity}/{run.project}/{data_path}:latest",
            type="dataset",
        )
        artifact.download(
            root=str(Path.cwd()),
        )
        self.filenames_list = [fp.name for fp in self.data_path.glob("*.json")]
        if config["n_samples"] is not None:
            self.filenames_list = self.filenames_list[: config["n_samples"]]
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        filename = self.filenames_list[idx]
        local_data_path = self.data_path / filename
        if local_data_path.is_file():
            with local_data_path.open("r") as f:
                data = json.load(fp=f)

        output = self.transform(
            data=data,
        )
        # fmt: off
        return {
            k: v
            for k, v
            in output.items()
            if k not in ["token_list", "token_word_map"]
        }
        # fmt: on





# Fonction de création des données à partir des données brutes

from time import perf_counter
def dataset_creator(folder: Path):
  bbox_folder = folder / 'box'
  entities_folder = folder / 'entities'
  img_folder = folder / 'img'

  # Sort by filename so that when zipping them together
  # we don't get some other file (just in case)
  entities_files = sorted(entities_folder.glob("*.txt"))
  bbox_files = sorted(bbox_folder.glob("*.txt"))
  img_files = sorted(img_folder.glob("*.jpg"))

  data = []

  print("Reading dataset:")
  for bbox_file, entities_file, img_file in tqdm(zip(bbox_files, entities_files, img_files), total=len(bbox_files)):            
    # Read the files
    bbox = read_bbox_and_words(bbox_file)
    entities = read_entities(entities_file)
    image = Image.open(img_file)

    # Assign labels to lines in bbox using entities
    bbox_labeled = assign_labels(bbox, entities)
    del bbox

    # Split lines into separate tokens
    new_bbox_l = []
    for index, row in bbox_labeled.iterrows():
      new_bbox_l += split_line(row)
    new_bbox = pd.DataFrame(new_bbox_l, columns=bbox_labeled.columns, dtype=np.int16)
    del bbox_labeled


    # Do another label assignment to keep the labeling more precise 
    for index, row in new_bbox.iterrows():
      label = row['label']

      if label != "O":
        entity_values = entities.iloc[0, entities.columns.get_loc(label.lower())]
        entity_set = entity_values.split()
        
        if any(SequenceMatcher(a=row['line'], b=b).ratio() > 0.7 for b in entity_set):
            label = "S-" + label
        else:
            label = "O"
      
      new_bbox.at[index, 'label'] = label

    width, height = image.size
  
    data.append([new_bbox, width, height])

  return data

dataset_train = dataset_creator('data_loading/SROIE' / 'train')
dataset_test = dataset_creator('data_loading/SROIE' / 'test')



def write_dataset(dataset: list, output_dir: Path, name: str):
  print(f"Writing {name}ing dataset:")
  with open(output_dir / f"{name}.txt", "w+", encoding="utf8") as file, \
       open(output_dir / f"{name}_box.txt", "w+", encoding="utf8") as file_bbox, \
       open(output_dir / f"{name}_image.txt", "w+", encoding="utf8") as file_image:

      # Go through each dataset
      for datas in tqdm(dataset, total=len(dataset)):
        data, width, height = datas
        
        filename = data.iloc[0, data.columns.get_loc('filename')]

        # Go through every row in dataset
        for index, row in data.iterrows():
          bbox = [int(p) for p in row[['x0', 'y0', 'x2', 'y2']]]
          normalized_bbox = normalize(bbox, width, height)

          file.write("{}\t{}\n".format(row['line'], row['label']))
          file_bbox.write("{}\t{} {} {} {}\n".format(row['line'], *normalized_bbox))
          file_image.write("{}\t{} {} {} {}\t{} {}\t{}\n".format(row['line'], *bbox, width, height, filename))

        # Write a second newline to separate dataset from others
        file.write("\n")
        file_bbox.write("\n")
        file_image.write("\n")
        
dataset_directory = Path('dataset')

dataset_directory.mkdir(parents=True, exist_ok=True)

write_dataset(dataset_train, dataset_directory, 'train')
write_dataset(dataset_test, dataset_directory, 'test')

# Creating the 'labels.txt' file to the the model what categories to predict.

labels = ['COMPANY', 'DATE', 'ADDRESS', 'TOTAL']
IOB_tags = ['S']
with open(dataset_directory / 'labels.txt', 'w') as f:
  for tag in IOB_tags:
    for label in labels:
      f.write(f"{tag}-{label}\n")
  # Writes in the last label O - meant for all non labeled words
  f.write("O")
