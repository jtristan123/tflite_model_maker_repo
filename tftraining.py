import tensorflow as tf
import pathlib 

#cones_pic = tf.keras.utils.get_file('images_and_ann0_OG') #only typically used for downloading files

cones_pic = pathlib.Path('images_and_anno_OG')
cones_pic

list_dataset = tf.data.Dataset.list_files(str(cones_pic/'*.jpg'))

for file in list_dataset.take(5):
    print(file.numpy())

#split images into training and validation sets
train_dataset = list_dataset.take(10)
val_dataset = list_dataset.skip(80)
print("Training dataset:")

for file in train_dataset:
    print(file.numpy())