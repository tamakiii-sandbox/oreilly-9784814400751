import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    NUM_CLASSES = 10

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = y_train.astype('float32') / 255.0
    y_test = y_test.astype('float32') / 255.0

    # n * (3072+1) = 614,600 // (n = number of nodes, 200 for instance)
    model = models.Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(200, activation = 'relu'),
      layers.Dense(250, activation = 'relu'),
      layers.Dense(10, activation = 'softmax'),
    ])

    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(200, activation = 'relu')(x)
    x = layers.Dense(150, activation = 'relu')(x)
    output_layer = layers.Dense(10, activation = 'softmax')(x)
    model = models.Model(input_layer, output_layer)

    model.summary()

    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size = 32, epochs = 10, shuffle = True)

if __name__ == "__main__":
    main()
