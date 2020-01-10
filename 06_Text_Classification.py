import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

#print(test_data[1])

word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"]=0
word_index["<START>"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Make a fix length text
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[1]))
#print(len(decode_review(test_data[0])), len(decode_review(test_data[1])))

"""
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print("Results %", results)

model.save("save_models/text_classification_model.h5")
"""

model = keras.models.load_model("save_models/text_classification_model.h5")

"""
test_review = test_data[0]
prediction = model.predict([test_review])
print("--Review--")
print(decode_review(test_review))
print("Prediction: ", str(prediction[0]))
print("Actual: ", str(test_labels[0]))
"""

def review_encode(text):
    encoded = [1]

    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

with open("dataset/review.txt") as f:
    for line in f.readlines():
        nline = line.replace(","," ").replace("."," ").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)

        prediction = model.predict(encode)
        print("--Review--")
        print(encode)
        print(line)
        print("Prediction: ", str(prediction[0]))
        #print("Actual: ", str(test_labels[0]))
