from tflite_model_maker.object_detector import DataLoader, ObjectDetector
from tflite_model_maker import model_spec

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
