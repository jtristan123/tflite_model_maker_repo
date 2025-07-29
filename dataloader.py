from tflite_model_maker.object_detector import DataLoader

train_data = DataLoader.from_pascal_voc(
    images_dir='images',
    annotations_dir='images',
    label_map={1: 'cone'}
)
