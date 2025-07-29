from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
import tensorflow as tf
from absl import logging

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

DATA_DIR = 'images'

# ✅ Fixed label_map and model_spec
train_data = object_detector.DataLoader.from_pascal_voc(
    images_dir=DATA_DIR,
    annotations_dir=DATA_DIR,
    label_map={1: 'cone'}
)

model = object_detector.create(
    train_data=train_data,
    model_spec=model_spec.get('efficientdet_lite0'),  # ✅ this is the right way
    batch_size=8,
    train_whole_model=True,
    epochs=50,
    validation_data=train_data
)

model.export(
    export_dir='exported-model3',
    export_format=[ExportFormat.TFLITE, ExportFormat.LABEL]
)

print("✓ Training and export complete. Files saved to ./exported-model3")
