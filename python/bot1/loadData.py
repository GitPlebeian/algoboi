import json
import os

# Get the directory of the current python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
json_path = os.path.join(current_dir, '..', '..', 'shared', 'test.json')

# Read and load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

print(data["closes"])


# with open("././shared/test.json", "r") as json_file:
# 	data = json.load(json_file)

# 	print(data)