from tflite_model_maker.object_detector import DataLoader

# Adjust the path if your images folder is somewhere else
train_data = DataLoader.from_pascal_voc(
    'images', 'images', ['cone']
)

print("✓ Loaded", len(train_data), "images.")