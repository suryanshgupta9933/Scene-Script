print('Starting Preprocessing...')

# Importing the libraries
import argparse

# Take arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--caption_file', type=str, help='Path to Captions File')
args = parser.parse_args()

# Defining Paths
caption_file = args.caption_file
print('Paths: ' + '\u2713')

# Loading caption file
with open(caption_file, 'r', encoding='utf8') as f:
    captions = f.readlines()

    new_lines = []
    for i, line in enumerate(captions):
        if i == 0:
            continue
        parts = line.strip().split(',')
        if len(parts) == 3:
            image_file, comment, caption = parts
            new_lines.append(image_file + ',' + caption + '\n')

    with open(caption_file, 'w') as f:
        f.writelines(new_lines)

print('Preprocessing: ' + '\u2713')