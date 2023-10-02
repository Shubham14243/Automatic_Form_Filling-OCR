import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tfd
import matplotlib.pyplot as plt
from efficientnet.model import layers
from keras.utils.version_utils import callbacks
from tensorflow import keras

IMG_WIDTH = 200
IMG_HEIGHT = 50
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 16
EPOCHS = 100
MODEL_NAME = 'Text-OCR'
IMAGE = 'img'

CALLBACKS = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath=MODEL_NAME + ".h5", save_best_only=True)
]

LEARNING_RATE = 1e-3
np.random.seed(2569)
tf.random.set_seed(2569)

train_csv_path = './CSV/written_name_train.csv'
valid_csv_path = './CSV/written_name_validation.csv'
test_csv_path = './CSV/written_name_test.csv'

train_image_dir = './train_v2/train'
valid_image_dir = './validation_v2/validation'
test_image_dir = './test_v2/test/'

TRAIN_SIZE = BATCH_SIZE * 1000
VALID_SIZE = BATCH_SIZE * 500
TEST_SIZE  = BATCH_SIZE * 100

AUTOTUNE = tfd.AUTOTUNE

train_csv = pd.read_csv(train_csv_path)[:TRAIN_SIZE]
valid_csv = pd.read_csv(valid_csv_path)[:VALID_SIZE]
test_csv = pd.read_csv(test_csv_path)[:TEST_SIZE]
train_csv.head()

train_labels = [str(word) for word in train_csv['IDENTITY'].to_numpy()]
train_labels[:10]
unique_chars = set(char for word in train_labels for char in word)
n_classes = len(unique_chars)
print(f"Total number of unique characters : {n_classes}")
print(f"Unique Characters : \n{unique_chars}")

MAX_LABEL_LENGTH = max(map(len, train_labels))
print(f"Maximum length of a label : {MAX_LABEL_LENGTH}")

train_csv['FILENAME'] = [train_image_dir + f"/{filename}" for filename in train_csv['FILENAME']]
valid_csv['FILENAME'] = [valid_image_dir + f"/{filename}" for filename in valid_csv['FILENAME']]
test_csv['FILENAME']  = [test_image_dir + f"/{filename}" for filename in test_csv['FILENAME']]
train_csv.head()

char_to_num = layers.StringLookup(
    vocabulary = list(unique_chars),
    mask_token = None
)

num_to_char = layers.StringLookup(
    vocabulary = char_to_num.get_vocabulary(),
    mask_token = None, 
    invert = True
)

def load_image(image_path : str):
    image = tf.io.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(contents = image, channels = 1)
    cnvt_image = tf.image.convert_image_dtype(image = decoded_image, dtype = tf.float32)
    resized_image = tf.image.resize(images = cnvt_image, size = (IMG_HEIGHT, IMG_WIDTH))

    image = tf.transpose(resized_image, perm = [1, 0, 2])
    image = tf.cast(image, dtype = tf.float32)
    return image

def encode_single_sample(image_path : str, label : str):
    image = load_image(image_path)
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    vecs = char_to_num(chars)

    pad_size = MAX_LABEL_LENGTH - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings = [[0, pad_size]], constant_values=n_classes+1)
    return {'image':image, 'label':vecs}

train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(train_csv['FILENAME'].to_list()), np.array(train_csv['IDENTITY'].to_list()))
).shuffle(1000).map(encode_single_sample, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(valid_csv['FILENAME'].to_list()), np.array(valid_csv['IDENTITY'].to_list()))
).map(encode_single_sample, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(test_csv['FILENAME'].to_list()), np.array(test_csv['IDENTITY'].to_list()))
).map(encode_single_sample, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print(f"Training Data Size   : {tf.data.Dataset.cardinality(train_ds).numpy() * BATCH_SIZE}")
print(f"Validation Data Size : {tf.data.Dataset.cardinality(valid_ds).numpy() * BATCH_SIZE}")
print(f"Testing Data Size    : {tf.data.Dataset.cardinality(test_ds).numpy() * BATCH_SIZE}")

def show_images(data, GRID=[4,4], FIGSIZE=(25, 8), cmap='binary_r', model=None, decode_pred=None):
    plt.figure(figsize=FIGSIZE)
    n_rows, n_cols = GRID

    data = next(iter(data))
    images, labels = data['image'], data['label']

    for index, (image, label) in enumerate(zip(images, labels)):
        text_label = num_to_char(label)
        text_label = tf.strings.reduce_join(text_label).numpy().decode('UTF-8')
        text_label = text_label.replace("[UNK]", " ").strip()
        plt.subplot(n_rows, n_cols, index+1)
        plt.imshow(tf.transpose(image, perm=[1,0,2]), cmap=cmap)
        plt.axis('off')
        
        if model is not None and decode_pred is not None:
            pred = model.predict(tf.expand_dims(image, axis=0))
            pred = decode_pred(pred)[0]
            title = f"True : {text_label}\nPred : {pred}"
            plt.title(title)
        else:
            plt.title(text_label)
        plt.show()

show_images(data=train_ds, cmap='gray')
from tensorflow.keras.utils import register_keras_serializable
@register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_len = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_len = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred

input_images = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image")
target_labels = layers.Input(shape=(None, ), name="label")

x = layers.Conv2D(
    filters=32, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(input_images)
x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(
    filters=64, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(x)
x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
encoding = layers.Reshape(target_shape=((IMG_WIDTH//4), (IMG_HEIGHT//4)*64))(x)
encoding = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(encoding)
encoding = layers.Dropout(0.2)(encoding)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(encoding)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
output = layers.Dense(len(char_to_num.get_vocabulary())+1, activation='softmax')(x)
ctc_layer = CTCLayer()(target_labels, output)

ocr_model = keras.Model(
    inputs=[input_images, target_labels],
    outputs=[ctc_layer]
)
ocr_model.summary()
tf.keras.utils.plot_model(ocr_model)
ocr_model.compile(optimizer='adam')

history = ocr_model.fit(
    train_ds, 
    validation_data=valid_ds, 
    epochs=EPOCHS,
    callbacks=CALLBACKS
)

from tensorflow import keras
keras.utils.custom_object_scope
with keras.utils.custom_object_scope({'CTCLayer': CTCLayer}):
    ocr_model = keras.models.load_model('model_1.h5')
inference_model = keras.Model(
    inputs=ocr_model.get_layer(name="image").input,
    outputs=ocr_model.get_layer(name='dense_1').output
)
inference_model.summary()

def decode_pred(pred_label):
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:,:MAX_LABEL_LENGTH]
    chars = num_to_char(decode)
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
    return filtered_texts

print(decode_pred(inference_model.predict(test_ds)))
#show_images(data=test_ds, model=inference_model, decode_pred=decode_pred, cmap='binary')
#show_images(data=valid_ds, model=inference_model, decode_pred=decode_pred, cmap='binary')

