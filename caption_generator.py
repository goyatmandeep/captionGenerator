# -*- coding: utf-8 -*-
BASE = "/caption-generator/"

"""**PART 1**
Extracting features from the images.
"""
# import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dropout, Dense, LSTM, Embedding
from keras.models import Model
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import os
import pickle
import string
from collections import Counter
import gc
from keras.preprocessing.text import Tokenizer

def get_image_features(directory):
  image_features = {}
  # in_layer = Input(shape=(224, 224, 3))
  model = VGG16()
  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  # model.summary()
  count = 0
  for image_name in os.listdir(directory):
    image_path = os.path.join(directory, image_name)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    image_feature = model.predict(image, verbose=0)
    image_name = image_name.split('.')[0]
    image_features[image_name] = image_feature[0]
    count += 1
    if count%500 == 0:
      print(count)
  return image_features

image_dataset_directory = "/Flickr_Data/Flickr_Data/Images"
image_features = get_image_features(image_dataset_directory)
print(len(image_features))
with open(BASE+"image_features.p", "wb") as file:
  pickle.dump(image_features, file)

"""**PART 2**
Textual data processing begins.
"""

# import pickle
OOV_token = "oov"
start_token = "startcap"
end_token = "endcap"
MIN_OCCUR = 2
MAX_LEN = 24  #22 + 2 for start and end token

with open(BASE+"image_features.p", "rb") as file:
  image_features = pickle.load(file)
print(len(image_features))

def reduce_vocabulary(image_captions, count):
  maxlen = MAX_LEN
  avg = total = 0
  for image_id, captions in image_captions.items():
    for i, caption in enumerate(captions):
      caption = caption.split()
      caption = [word for word in caption if count[word] >= MIN_OCCUR]
      # maxlen = max(maxlen, len(caption))
      # avg += len(caption)
      # total += 1
      end = min(len(caption)-1, maxlen-1)
      image_captions[image_id][i] = ' '.join(caption[:end])+' '+end_token
  # print(avg/total)
  return image_captions, maxlen

def clean_image_captions(image_captions, count=None):
  ''' image_captions : dict : image_id (str), captions (list of distinct captions)'''
  table = str.maketrans('', '', string.punctuation)
  for image_id, captions in image_captions.items():
    for i, caption in enumerate(captions):
      caption = caption.split()
      #remove punctuations
      caption = [word.translate(table) for word in caption] 
      #make it lower
      caption = [word.lower() for word in caption]
      #remove single chars
      caption = [word for word in caption if len(word)>1]
      #remove numbers
      caption = [word for word in caption if word.isalpha()]

      image_captions[image_id][i] = start_token + " " + ' '.join(caption)+" "+end_token
      count.update(caption)
      count.update([start_token, end_token])
  if count:
    return image_captions, count
  return image_captions

def get_image_captions(file_addr):
  image_captions = {}
  with open(file_addr, "r") as file:
    for line in file:
      line = line.split()
      image_id, image_caption = line[0].split('.')[0], line[1:]
      if image_id not in image_captions:
        image_captions[image_id] = []
      image_captions[image_id].append(' '.join(image_caption))
  # print(len(image_captions))
  return image_captions

caption_dataset_location = "/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
image_captions = get_image_captions(caption_dataset_location)
image_captions, count = clean_image_captions(image_captions, Counter())
for key, value in image_captions.items():
  print(key, value)
  break
image_captions, maxlen = reduce_vocabulary(image_captions, count)
temp = 0
for key, value in image_captions.items():
  print(key, value)
  temp += 1
  if temp == 10:
    break
print(len(image_captions))
with open(BASE+"image_captions.p", "wb") as file:
  pickle.dump(image_captions, file)

print(maxlen)

"""**PART 3** Final Data Preparation before defining the model."""

OOV_token = "oov"

with open(BASE+"image_captions.p", "rb") as file:
  image_captions = pickle.load(file)

def get_tokenizer(location, image_captions):
  temp = []
  with open(location, "r") as file:
    for line in file:
      image_id = line.strip().split('.')[0]
      try:
        temp = temp+image_captions[image_id]
      except:
        print(f"File missing {image_id}")
  tokenizer = Tokenizer(oov_token=OOV_token)
  tokenizer.fit_on_texts(temp)
  return tokenizer

train_captions_location = "/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
tokenizer = get_tokenizer(train_captions_location, image_captions)
print(len(tokenizer.word_index))
with open(BASE+"tokenizer.p", "wb") as file:
  pickle.dump(tokenizer, file)
gc.collect()

"""**PART 4** Deifining model and data generator"""

# maxlen = 24

# with open(BASE+"image_captions.p", "rb") as file:
#   image_captions = pickle.load(file)
# with open(BASE+"image_features.p", "rb") as file:
#   image_features = pickle.load(file)
# with open(BASE+"tokenizer.p", "rb") as file:
#   tokenizer = pickle.load(file)

def data_generator(tokenizer, image_features, image_captions, file_location, maxlen):
  while True:
    with open(file_location, "r") as file:
      for line in file:
        image_id = line.strip().split('.')[0]
        X1, X2, Y = [], [], []
        image = image_features[image_id]
        for caption in image_captions[image_id]:
          caption = tokenizer.texts_to_sequences([caption])[0]
          for i in range(1, len(caption)):
            in_seq, out_seq = caption[:i], caption[i]
            in_seq = pad_sequences([in_seq], maxlen=maxlen, truncating="post")[0]
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index)+1)[0]
            X1.append(image)
            X2.append(in_seq)
            Y.append(out_seq)
        yield [[np.array(X1), np.array(X2)], np.array(Y)]

def training_data(tokenizer, image_features, image_captions, file_location, maxlen):
  X1, X2, Y = [], [], []
  with open(file_location, "r") as file:
      for line in file:
        image_id = line.strip().split('.')[0]
        image = image_features[image_id]
        for caption in image_captions[image_id]:
          caption = tokenizer.texts_to_sequences([caption])[0]
          for i in range(1, len(caption)):
            in_seq, out_seq = caption[:i], caption[i]
            in_seq = pad_sequences([in_seq], maxlen=maxlen)[0]
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index)+1)[0]
            X1.append(image)
            X2.append(in_seq)
            Y.append(out_seq)
  return np.array(X1), np.array(X2), np.array(Y)

vocab_embedd = {}
for word, index in tokenizer.word_index.items():
  vocab_embedd[word] = None
with open("/Glove/glove.6B.300d.txt", "r") as file:
  for line in file:
    line = line.strip().split()
    if line[0] == '.':
      vocab_embedd[end_token] = np.array(line[1:], dtype=float)
    elif line[0] in vocab_embedd:
      vocab_embedd[line[0]] = np.array(line[1:], dtype=float)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
for word, i in tokenizer.word_index.items():
  embedding_vector = vocab_embedd.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
  else:
    embedding_matrix[i] = np.random.uniform(low=-0.5, high=0.5, size=300)
vocab_embedd = None
gc.collect()

def get_model(vocab_size, maxlen):
  
  inputs1 = Input(shape=(4096,))
  image1 = Dropout(0.5)(inputs1)
  image2 = Dense(256, activation="relu")(image1)
  
  inputs2 = Input(shape=(maxlen,))
  feature1 = Embedding(vocab_size, 300, input_length=maxlen,  mask_zero=True, weights=[embedding_matrix], trainable=True)(inputs2)
  feature2 = Dropout(0.5)(feature1)
  feature3 = LSTM(256)(feature2)

  decoder1 = add([image2, feature3])
  decoder2 = Dense(256, activation="relu")(decoder1)
  output = Dense(vocab_size, activation="softmax")(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=output)
  model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(
      learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    )
  )
  model.summary()
  plot_model(model, to_file="/caption-generator/model.png", show_shapes=True)
  return model

model = get_model(len(tokenizer.word_index)+1, maxlen)

"""PART 5 Training the model (Run Part 4 First)"""

checkpoint = ModelCheckpoint("/caption-model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
epochs = 20
file_location = "/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
file_location2 = "/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt"
with open(file_location, "r") as file:
  steps = len(file.readlines())
with open(file_location2, "r") as file:
  steps2 = len(file.readlines())
# X1, X2, Y = training_data(tokenizer, image_features, image_captions, file_location, maxlen)
# vX1, vX2, vY = training_data(tokenizer, image_features, image_captions, file_location2, maxlen)
# history = model.fit([X1, X2], Y, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=([vX1, vX2], vY))
train_generator = data_generator(tokenizer, image_features, image_captions, file_location, maxlen)
validation_generator = data_generator(tokenizer, image_features, image_captions, file_location2, maxlen)
history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1, validation_data=validation_generator, validation_steps=steps2, validation_freq=1, callbacks=[checkpoint])

print(history.history.keys())

# acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

# plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.title('Training accuracy')
# plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')

plt.figure()
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Validation Loss")

plt.legend()

plt.show()
with open("history.p", "wb") as file:
  pickle.dump(history.history, file)

"""**PART 5** Evaluation on Test Set"""

def tokenizer_reverse(tokenizer):
  mapping = tokenizer.word_index
  reverse_mapping = {}
  for word, index in mapping.items():
    reverse_mapping[index] = word
  return reverse_mapping

from numpy import argmax
def generate_caption(model, tokenizer, feature, maxlen, end_token):
  in_seq = [tokenizer.word_index["startcap"]]
  padded_in_seq = pad_sequences([in_seq], maxlen=maxlen)[0]
  # print(padded_in_seq)
  caption = []
  for i in range(maxlen):
    # print(in_seq)
    next_word = model.predict([[feature], [padded_in_seq]])[0]
    next_word = argmax(next_word)
    in_seq.append(next_word)
    if next_word == end_token:
      break
    padded_in_seq = np.array(pad_sequences([in_seq], maxlen=maxlen)[0])
  if in_seq[-1] != end_token:
    in_seq.append(end_token)
  return in_seq

def evaluate(model, tokenizer, image_features, image_captions, file_location, maxlen):
  generated = []
  reference = []
  mapping = tokenizer_reverse(tokenizer)
  end_token = tokenizer.word_index["endcap"]
  with open(file_location, "r") as file:
    count = 0
    for line in file:
      image_id = line.split('.')[0]
      in_seq = generate_caption(model, tokenizer, image_features[image_id], maxlen, end_token)
      in_seq = [mapping[index] for index in in_seq]
      generated.append(in_seq)
      if count < 5:
        img = Image.open("/Flickr_Data/Flickr_Data/Images/"+line.strip())
        plt.figure()
        plt.imshow(img)
        plt.show()
        print(in_seq)
      reference.append([list(caption.split()) for caption in image_captions[image_id]])
      count += 1
  print(f'BLEU-1: {corpus_bleu(reference, generated, weights=(1.0, 0, 0, 0))}')
  print(f'BLEU-2: {corpus_bleu(reference, generated, weights=(0.5, 0.5, 0, 0))}')
  print(f'BLEU-3: {corpus_bleu(reference, generated, weights=(0.3, 0.3, 0.3, 0))}')
  print(f'BLEU-4: {corpus_bleu(reference, generated, weights=(0.25, 0.25, 0.25, 0.25))}')

file_location3 = "/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt"
evaluate(model, tokenizer, image_features, image_captions, file_location3, maxlen)

"""PART 6 Generate Captions of Personal Images"""

# import tensorflow as tf

def get_image_features(directory):
  image_features = {}
  # in_layer = Input(shape=(224, 224, 3))
  model = VGG16()
  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  # model.summary()
  count = 0
  for image_name in os.listdir(directory):
    image_path = os.path.join(directory, image_name)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    image_feature = model.predict(image, verbose=0)
    image_name = image_name.split('.')[0]
    image_features[image_name] = image_feature[0]
  return image_features

# with open(BASE+"tokenizer.p", "rb") as file:
  # tokenizer = pickle.load(file)
directory = "/caption-generator/sample-images/"
mapping = tokenizer_reverse(tokenizer)
sample_image_features = get_image_features(directory)
# maxlen = 24
sample_image_captions = {}
# model = load_model("caption-model.h5")
for image_id, feature in sample_image_features.items():
  caption = generate_caption(model, tokenizer, feature, maxlen, tokenizer.word_index["endcap"])
  sample_image_captions[image_id] = ' '.join([mapping[id] for id in caption[1:-1]])

for filename in os.listdir(directory):
    img = Image.open(directory+filename)
    plt.figure()
    plt.imshow(img)
    plt.show()
    print(sample_image_captions[filename.split('.')[0]])