import pickle
import os
import json

# python keyword_to_list \path\to\keyword.json

data_path = os.sys.argv[1]
output_path = os.sys.argv[2] if len(os.sys.argv) > 2 else 'keyword.json'

keyword_list = []

with open(data_path, 'r') as f:
  data = json.load(f)
  for k, v in data.items():
    for keyword in v:
      if keyword not in keyword_list:
        keyword_list.append(keyword)

with open(output_path, 'w') as f:
  json.dump({'keyword': keyword_list}, f)
