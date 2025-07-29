from tflite_model_maker.object_detector import DataLoader, ObjectDetector
from tflite_model_maker import model_spec
from tflite_model_maker.config import ExportFormat


train_data = DataLoader.from_pascal_voc(
    'images',  # images folder
    'images',  # annotations folder
    ['cone']   # class list
)

spec = model_spec.get('efficientdet_lite0')

model = ObjectDetector.create(
    train_data=train_data,
    model_spec=spec,
    batch_size=8,
    train_whole_model=True,
    epochs=50,
    validation_data=None
)

model.export(
    export_dir='exported-model',
    export_format=[ExportFormat.TFLITE, ExportFormat.LABEL]
)
print("✓ Training and export complete. Files saved to ./exported-model")