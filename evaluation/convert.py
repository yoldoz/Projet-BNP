import pandas as pd
df = pd.read_json ('Path where the JSON file is saved\File Name.json')
df.to_csv ('Path where the new TEXT file will be stored\New File Name.txt', index = False)