# Script de chargement et du preprocessing du dataset SROIE
# En sortie des fichiers textes prêts à être donnés au modèle 

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



# Etape 1 : Fonction de lecture des fichiers "box" 

sroie_folder_path = Path('SROIE')
example_file = Path('X51005365187.txt')

def read_bbox_and_words(path: Path):
  bbox_and_words_list = []

  with open(path, 'r', errors='ignore') as f:
    for line in f.read().splitlines():
      if len(line) == 0:
        continue
        
      split_lines = line.split(",")

      bbox = np.array(split_lines[0:8], dtype=np.int32)
      text = ",".join(split_lines[8:])

      # From the splited line we save (filename, [bounding box points], text line).
      # The filename will be useful in the future
      bbox_and_words_list.append([path.stem, *bbox, text])
    
  dataframe = pd.DataFrame(bbox_and_words_list, columns=['filename', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'line'], dtype=np.int16)
  dataframe = dataframe.drop(columns=['x1', 'y1', 'x3', 'y3'])

  return dataframe

# Exemple de chargement d'une box

#bbox_file_path = sroie_folder_path /+ "test/box" +/ example_file
#print("== File content ==")
#!head -n 5 "{bbox_file_path}"

#bbox = read_bbox_and_words(path=bbox_file_path)
#print("\n== Dataframe ==")
#bbox.head(5)


# Etape 2 : Fonction de lecture des fichiers "entities" qui correspondent aux labels associés aux "box"

def read_entities(path: Path):
  with open(path, 'r') as f:
    data = json.load(f)

  dataframe = pd.DataFrame([data])
  return dataframe


# Exemple de chargement d'une entitie

#entities_file_path = sroie_folder_path /+  "test/entities" +/ example_file
#print("== File content ==")
#!head "{entities_file_path}"

#entities = read_entities(path=entities_file_path)
#print("\n\n== Dataframe ==")
#entities

# Etape 3 : Fonction d'assignation d'un label à chaque ligne (Company, Adress, Date, Total ou O pour rien)
# Entrée : Ligne d'un fichier box + entité associée au fichier
# Sortie : Label associé à la ligne 
# Fonctionnement : on regarde dans le fichier entité comment elle labelisée la ligne
# Si la ligne n'apparait pas dans le fichier entité, c'est un O  
    
def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    # Lit une ligne du fichier box et la sépare à chaque espace
    for i, column in enumerate(entities):
        # Pour chaque label dans entities on récupère la valeur ('F&P Pharmacy' ou '31.6€',...)
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()
        
        matches_count = 0
        for l in line_set:
            # Pour chaque élément de la ligne que l'on regarde
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                # On fait un test de similarité sur chaque élément, si >80% on compte 1 match
                matches_count += 1
            
            if (column.upper() == 'ADDRESS' and (matches_count / len(line_set)) >= 0.5) or \
               (column.upper() != 'ADDRESS' and (matches_count == len(line_set))) or \
               matches_count == len(entity_set):
                   # Si on trouve une similarité on retourne la label 
                return column.upper()

    return "O"


#line = bbox.loc[1,"line"]
#label = assign_line_label(line, entities)
#print("Line:", line)
#print("Assigned label:", label)


# Fonction qui permet d'assigner le Total et la Date qu'une seule fois (à la bounding box la plus large)
# et qui n'autorise pas l'assignation de l'adresse après la date ou le total (l'adresse est toujours au début)

def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
    already_labeled = {"TOTAL": False,
                       "DATE": False,
                       "ADDRESS": False,
                       "COMPANY": False,
                       "O": False
    }

    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words['line']):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if (label == "ADDRESS" and already_labeled["TOTAL"]) or \
           (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
            label = "O"

        # Assign to the largest bounding box
        if label in ["TOTAL", "DATE"]:
            x0_loc = words.columns.get_loc("x0")
            bbox = words.iloc[i, x0_loc:x0_loc+4].to_list()
            area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

            if max_area[label][0] < area:
                max_area[label] = (area, i)

            label = "O"

        labels.append(label)

    labels[max_area["DATE"][1]] = "DATE"
    labels[max_area["TOTAL"][1]] = "TOTAL"

    words["label"] = labels
    return words


# Example usage
#bbox_labeled = assign_labels(bbox, entities)
#bbox_labeled.head(15)

# Fonction de tokenisation 

def split_line(line: pd.Series):
  line_copy = line.copy()

  line_str = line_copy.loc["line"]
  words = line_str.split(" ")

  # Filter unwanted tokens
  words = [word for word in words if len(word) >= 1]

  x0, y0, x2, y2 = line_copy.loc[['x0', 'y0', 'x2', 'y2']]
  bbox_width = x2 - x0
  

  new_lines = []
  for index, word in enumerate(words):
    x2 = x0 + int(bbox_width * len(word)/len(line_str))
    line_copy.at['x0', 'x2', 'line'] = [x0, x2, word]
    new_lines.append(line_copy.to_list())
    x0 = x2 + 5 

  return new_lines


# Example usage
#new_lines = split_line(bbox_labeled.loc[1])
#print("Original row:")
#display(bbox_labeled.loc[1:1,:])

#print("Splitted row:")
#pd.DataFrame(new_lines, columns=bbox_labeled.columns)




# Normalisation des données

def normalize(points: list, width: int, height: int) -> list:
  x0, y0, x2, y2 = [int(p) for p in points]
  
  x0 = int(1000 * (x0 / width))
  x2 = int(1000 * (x2 / width))
  y0 = int(1000 * (y0 / height))
  y2 = int(1000 * (y2 / height))

  return [x0, y0, x2, y2]
