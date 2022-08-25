import csv, os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
      "--dataset",
      default="blended_skill_talk",
      type=str,
      help="dataset to train a chatbot. (huggingFace dataset/otter/simulator_dialo)",
    )

    parser.add_argument(
      "--dataset_path",
      default="../OTTer/data",
      type=str,
      help="path to data directory"
    )

    parser.add_argument(
      "--output_dir",
      default="./data",
      type=str,
      help="path to output directory."
    )

    parser.add_argument(
      "--output",
      default="output.csv",
      type=str,
      help="output file name."
    )

    parser.add_argument(
      "--keyword_path",
      default="./keyword.json",
      type=str,
      help="path to keyword file."
    )

    parser.add_argument(
      "--input",
      default="./output.jsonl",
      type=str,
      help="path to input dialogue file."
    )

    parser.add_argument("--split_number", default=7, type=int, 
      help="split dialogue t1,...tM to source t1,...tn and target tn+1,...,tM")

    args = parser.parse_args()

    return args

def parse_bst(args):
  dataset = load_dataset('blended_skill_talk')

  splits = ['train', 'validation']
  parse_datas = []

  key = json.load(open(args.keyword_path)) 
  keyword = key["keyword"]

  with open(os.path.join(args.output_dir, args.output), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['inputs', 'target'])
    for split in splits:
      for data in tqdm(dataset[split]):
        source_text = ''
        target_text = ''
        for cnt, (a, b) in enumerate(zip(data['free_messages'], data['guided_messages'])):
          if cnt < 4:
            source_text += f'{a}{b}'
          elif cnt == 4:
            source_text += f'{a}'
            target_text += f'{b}'
          else:
            target_text += f'{a}{b}'
        target_text = target_text[:-1]
        
        for key in keyword:
          if key in source_text or key in target_text:
            writer.writerow([source_text, target_text])
            break

def parse_otter(args):
  data_dir = os.path.join(args.dataset_path)
  domains = ["in_domain", "out_of_domain"]
  splits = ["train", "dev", "test"]

  all_source, all_target = [], []

  for domain in domains:
      for split in splits:
          file_path = os.path.join(data_dir, domain, split, 'text.csv')
          try:
            with open(file_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for cnt, row in enumerate(reader):
                    if row[0] == 'inputs' and row[1] == 'target': continue
                    all_source.append(row[0])
                    all_target.append(row[1])
          except:
            print("You need to execute write_csv.py to get text.csv first.")
            exit(0)

  output_path = os.path.join(args.output_dir, args.output)
  with open(output_path, 'w') as csvfile:
      fieldnames = ['inputs', 'target']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for (source, target) in zip(all_source, all_target):
          writer.writerow({"inputs": source, "target": target})

def parse_simulator_dialo(args):
  split_number = args.split_number
  output_path = os.path.join(args.output_dir, args.output)

  sim_dialo = list(open(args.input, 'r'))
  with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["inputs", "target"])
    for dialo in tqdm(sim_dialo):
      dialo = json.loads(dialo)
      source_text = ''.join(dialo["dialog"][:split_number])
      target_text = ''.join(dialo["dialog"][split_number:])
      writer.writerow([source_text, target_text])

def parse_dstc8(args):
    splits = ["train"]
    output_path = os.path.join(args.output_dir, args.output)
    with open(output_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["inputs", "target"])
        for split in splits:
            filenames = os.listdir(
                os.path.join(args.dataset_path, split))
            filenames.remove("schema.json")
            filenames.sort(key=lambda x : int(x[-8:-5]))
            for name in tqdm(filenames):
                file_name = os.path.join(args.dataset_path, split, name)
                dialogues = json.load(open(file_name))
                for dialogue in dialogues:
                    utterances = [turn['utterance'] for turn in dialogue['turns']]
                    if len(utterances) < 4: continue
                    for i in range(2, len(utterances)-1):
                        writer.writerow([(utterances[i-2] + ' ' + utterances[i-1] + ' ' + utterances[i]), 
                                          utterances[i+1]])


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "blended_skill_talk":
      parse_bst(args)
    elif args.dataset == "otter":
      parse_otter(args)
    elif args.dataset == "simulator_dialo":
      parse_simulator_dialo(args)
    elif args.dataset == "dstc8":
      parse_dstc8(args)
