import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import datasets, utils
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 10

def main():
    input_layer = Input((32,32,3))

    x = Conv2D(filters=32, kernel_size=3 , strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)

    output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input_layer, output_layer)
    model.summary()

    data = datasets.cifar10.load_data()
    x_test, y_test = evaluate(data, model)
    show(x_test, y_test, model)

def evaluate(data, model):
    (x_train, y_train), (x_test, y_test) = data

    # Normalize the image data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    # Compile the model
    opt = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test)

    # Return preprocessed x_test and y_test for visualization
    return x_test, y_test

def show(x_test, y_test, model):
    CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    preds = model.predict(x_test)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    actual_single = CLASSES[np.argmax(y_test, axis=-1)]

    n_to_show = 10
    indices = np.random.choice(len(x_test), n_to_show, replace=False)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(indices):
        img = x_test[idx]

        ax = fig.add_subplot(1, n_to_show, i+1)
        ax.axis('off')
        ax.text(0.5, -0.35, f'pred: {preds_single[idx]}', fontsize=10, ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.70, f'act: {actual_single[idx]}', fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img)

    # Show the figure
    plt.show()

def cnn_sample():
    input_layer = Input(shape=(32, 32, 3))
    conv_layer_1 = Conv2D(filters=10, kernel_size=(4, 4), strides=2, padding='same')(input_layer)
    conv_layer_2 = Conv2D(filters=20, kernel_size=(3, 3), strides=2, padding='same')(conv_layer_1)

    flatten_layer = Flatten()(conv_layer_2)
    output_layer = Dense(NUM_CLASSES, activation='softmax')(flatten_layer)
    model = Model(input_layer, output_layer)

    model.summary()

if __name__ == "__main__":
    main()
