from glob import glob

images_folder = '/home/user/DiffPose/downloads/images'

image_files = glob(f'{images_folder}/*/*.jpg')

# create and store csv with image's folder name, file_path and an id (sequential)
import pandas as pd

df = pd.DataFrame(image_files, columns=['file_path'])
df['folder'] = df['file_path'].apply(lambda x: x.split('/')[-2])
df['id'] = range(len(df))

df.to_csv('results/data.csv', index=False)

