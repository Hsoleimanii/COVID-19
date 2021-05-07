
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import os


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def augment_images(folder_sorce,folder_destination, num_augment):
  folder=folder_sorce
  count=0
  for filename in os.listdir(folder):
    img = load_img(os.path.join(folder,filename))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 256, 256)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=folder_destination, save_prefix=str(count), save_format='jpeg'):
        i += 1
        count+=1
        if i > num_augment:
            break  # otherwise the generator would loop indefinitely
  return
  
 
 def plot_costs(history,title):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title(title+': Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title(title+': Training and Validation Loss')
  plt.show()



####### functions for evaluations

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def compute_measures(cm,class_indx):

  TP = cm[class_indx][class_indx]
  FP=0
  for i in range(0,3):
    if i!=class_indx:
      FP=FP+cm[i][class_indx]

  FN=0
  for i in range(0,3):
    if i!=class_indx:
      FN=FN+cm[class_indx][i]

  TN=0
  for i in range(0,3):
      if i!=class_indx:
        TN=TN+cm[i][i]



  total = 0
  for i in range(0,3):
    for j in range(0,3):
      total=total+cm[i][j]


  accuracy = (TP + TN)/total  # accuracy is just the average accuracy of the model

  misclassification_rate = (FP+FN)/total
  true_positive = (TP)/(TP+FN)
  false_positive = (FP)/(TN+FP)
  precision = TP/(TP+FP)
  F1_score = 2*precision*true_positive/(precision+true_positive)
  specificity = TN/(TN+FN)

  print('accuracy = ', accuracy)
  print('Misclassification Rate = ', misclassification_rate)
  print('True positive rate = ', true_positive)
  print('False positive rate = ', false_positive)
  print('F1 Score = ', F1_score)
  print('Specificity = ', specificity)
  
  

