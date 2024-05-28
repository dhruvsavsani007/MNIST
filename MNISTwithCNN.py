import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras import metrics 
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
single_image = x_train[0]
print(single_image.shape)
print(single_image)
plt.imshow(single_image)
plt.show()
# print(y_train)
# print(y_train.shape)
y_example = to_categorical(y_train)
# print(y_example.shape)
# print(single_image.max(), single_image.min())
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)
x_train = x_train / 255
x_test = x_test / 255
# scaled_imange = x_train[0]
# plt.imshow(scaled_imange)
# plt.show()

# batch size, width, height, color channels
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="valid",
        input_shape=(28, 28, 1),
        activation="relu",
    )
)
y = model.add(MaxPool2D(pool_size=(2, 2)))

y = plt.imread(y)
plt.imshow(y)
plt.show()

model.add(Flatten())
model.add(Dense(128, activation="relu"))


# output layer activation func softmax beacause of multiclass problem
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=1)
# hist = model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stopping])

# metrics = pd.DataFrame(hist.history)
# metrics.to_csv('MNISTwithCNN.csv', index=False)
# model.save('MNISwithCNN.h5')

metrics = pd.read_csv('MNIST\MNISTwithCNN.csv')
print(metrics)
# metrics[['loss', 'val_loss']].plot()
# plt.show()
# metrics[['accuracy', 'val_accuracy']].plot()
# plt.show()

model_later = load_model("MNIST\MNISwithCNN.h5")
# print(model_later.metrics_names)
prediction = model_later.predict(x_test)
# print(prediction)
temp = []
for i in prediction:
    temp.append(np.argmax(i))

# print(temp)
prediction = to_categorical(temp)
# print(prediction)
# print(prediction.shape)
# print(y_cat_test.shape)

dict = {'Actual':y_test, 'Predict':temp}
# compare = pd.DataFrame(dict)
# print(compare.sample(10))

# print(classification_report(y_test, temp))
# print(confusion_matrix(y_test, temp))

# sns.heatmap(confusion_matrix(y_test, temp), annot=True)
# plt.show()

image = x_test[0]
# plt.imshow(image)
# plt.show()

image = image.reshape(1, 28, 28, 1)
print(image.shape)
single_predict = np.argmax(model_later.predict(image))
print(single_predict)