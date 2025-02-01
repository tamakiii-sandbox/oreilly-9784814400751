import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    NUM_CLASSES = 10

    # Normalize the image data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    # Define the model using Functional API
    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(150, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input_layer, output_layer)

    model.summary()

    # Compile the model
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test)

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

if __name__ == "__main__":
    main()
