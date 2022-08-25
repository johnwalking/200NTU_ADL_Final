import json
import csv

with open('output.jsonl', 'r') as f:
    #json_list = list(json_file)
    json_list = [json.loads(line) for line in f]

with open('./keyword.json', 'r') as f:
    keys = json.load(f)["keyword"]

print(keys)
print(json_list[0])

numbers = []
for i in range(len(json_list)):
    tmp  = json_list[i]
    #print(tmp, type(tmp))

    det = False
    for line in tmp["dialog"]:
        if det ==True:
            break
        for key in keys:
            if key in line.split(" "):
                numbers.append(i)
                det = True
                break
print(numbers)
Input = []
Output = []

with open('simulator_valid.csv', 'w') as csvfile:
    fieldnames = ['inputs', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for num in numbers:
        cur_data = json_list[num]["dialog"]
        writer.writerow({"inputs": cur_data[0], "target": cur_data[1]})
        writer.writerow({"inputs": (cur_data[0] + "</s> <s>" + cur_data[1]), "target": cur_data[2]})
        for i  in range(3,len(cur_data)):
            in_seten = cur_data[i-3]+"</s> <s>"+cur_data[i-2]+"</s> <s>"+cur_data[i-1]
            out_seten = cur_data[i]
            writer.writerow({"inputs": in_seten, 'target': out_seten})





