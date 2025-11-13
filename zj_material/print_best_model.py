from pathlib import Path
import json5

models_path = Path("models_2")
assert models_path.exists() and models_path.is_dir(), "models directory not found"

print("Looking at", models_path, "directory for the best performing model so far...")
min_config = None
min_mse = 1e15
for file in models_path.iterdir():
    if file.is_file() and file.name.endswith(".json"):
        model = json5.load(open(file))
        if model['validation_mse'] < min_mse:
            min_mse = model['validation_mse']
            min_config = model

print("Best results obtained with", min_config)
print("The filename is", file.resolve())

