import os
import csv
from tqdm import tqdm

# python3 concat.py /path/to/data/dir /path/to/output/csv

data_dir = os.path.join(os.sys.argv[1])
domains = ["in_domain", "out_of_domain"]
splits = ["train", "dev", "test"]

all_source, all_target = [], []

for domain in domains:
    for split in splits:
        file_path = os.path.join(data_dir, domain, split, 'text.csv')
        with open(file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for cnt, row in enumerate(reader):
                if row[0] == 'inputs' and row[1] == 'target': continue
                all_source.append(row[0])
                all_target.append(row[1])

output_path = os.path.join(os.sys.argv[2]) # './valid_otters_data.csv'
with open(output_path, 'w') as csvfile:
    fieldnames = ['inputs', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for (source, target) in zip(all_source, all_target):
        writer.writerow({"inputs": source, "target": target})
