

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import os

from utils import augment_images,plot_costs , plot_confusion_matrix, compute_measures 


#  __________________________   augment the data and save it

augment_images('./data/Healthy/','./Augmented/Healthy/',10) 

augment_images('./data/Pneumonia/','./Augmented/Pneumonia/',10)

augment_images('./data/Covid-19/','./Augmented/Covid/',10)



#######  _______________________________ hyperparameters
img_height=256
img_width=256
batch_size=8
num_classes = 3



######## _______________________prepare training and validation   

data_dir='./Augmented/'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,   validation_split=0.2,
  subset="training",    seed=123,
  image_size=(img_height, img_width),   batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,   validation_split=0.2,
  subset="validation",   seed=123,
  image_size=(img_height, img_width),  batch_size=batch_size)



# visualize sample images from healthy and covide cases
class_names = train_ds.class_names
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(8):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    #plt.axis("off")



data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.02),
])


# ____________________________________  create the model
base_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(img_height,img_width,3), pooling=None)
base_model.trainable = False # freez it for now
# Let's take a look at the base model architecture
print(base_model.summary())

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes,activation = 'softmax')


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
])


base_learning_rate = 0.00005
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), #(from_logits=False)
              metrics=['accuracy'])
print(model.summary())


##########  ________________________________ train the model_____________
epochs=15
loss0, accuracy0 = model.evaluate(val_ds)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds)





plot_costs(history,'Model') 




# ___________________________________testing the model


from sklearn.metrics import classification_report, confusion_matrix
import itertools  

test_dir = './test/'
test_datagen= ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_dir,batch_size =1,class_mode ='categorical',target_size = (img_height,img_width)
,seed=10)



target_names = []
for key in test_generator.class_indices:
    target_names.append(key)





cm=np.zeros((3,3))
if 1:
  
  test_cov_dir= './test/Covid/'
  for filename in os.listdir(test_cov_dir):
    img = tf.keras.preprocessing.image.load_img(
      os.path.join(test_cov_dir,filename),target_size = (256,256))
    
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    image_array = tf.expand_dims(image_array,0)

    prediction  = model.predict(image_array)
    y_pred = np.argmax(prediction, axis=1)
    cm[0,y_pred]+=1

  print ('')
  print ('')

  test_cov_dir= './test/Healthy/'
  for filename in os.listdir(test_cov_dir):
    img = tf.keras.preprocessing.image.load_img(
      os.path.join(test_cov_dir,filename),target_size = (256,256))
    
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    image_array = tf.expand_dims(image_array,0)

    prediction  = model.predict(image_array)
    y_pred = np.argmax(prediction, axis=1)
    cm[1,y_pred]+=1



  print ('')
  print ('')

  test_cov_dir= './test/Pneumonia/'
  for filename in os.listdir(test_cov_dir):
    img = tf.keras.preprocessing.image.load_img(
      os.path.join(test_cov_dir,filename),target_size = (256,256))
    
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    image_array = tf.expand_dims(image_array,0)

    prediction  = model.predict(image_array)
    y_pred = np.argmax(prediction, axis=1)
    cm[2,y_pred]+=1

  print(cm)
  

plot_confusion_matrix(cm, target_names, title='Confusion MatriX:')
print('')
print('')
print('Classification measures: Covid')
compute_measures(cm,0)
print('')
print('')
print('Classification measures: Healthy')
compute_measures(cm,1)
print('')
print('')
print('Classification measures: Pneumonia')
compute_measures(cm,2)


