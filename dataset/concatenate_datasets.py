import json
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory containing the JSON files
data_dir = os.path.join(script_dir, 'data', 'processed')

# List to hold all combined data
combined_data = []

# Iterate through all JSON files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as file:
            try:
                data = json.load(file)
                # Assuming each JSON is a list of items, extend the combined list
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    # If it's a dict, append it as a single item
                    combined_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error loading {filename}: {e}")

# Write the combined data to a new JSON file in the script's directory
output_file = os.path.join(script_dir, 'combined_dataset.json')
with open(output_file, 'w') as file:
    json.dump(combined_data, file, indent=4)

print(f"Combined dataset saved to {output_file}")