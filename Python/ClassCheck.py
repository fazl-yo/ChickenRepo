import yaml

with open("data.yaml", "r") as file:
    data_config = yaml.safe_load(file)

class_names = data_config["names"]
print("Class names:", class_names)
