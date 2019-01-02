import os

from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate

def get_model(img_width, img_height, num_labeled_inputs, num_actions, weights_file_path=None):
    #Architecture based on Deep Mind https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    labeled_data_input = Input(shape=(num_labeled_inputs,), name='labeledInput')
    model = connect_layers(
        [
            Input(shape=(img_height,img_width,1), name='imgInput'),
            Conv2D(16, kernel_size=8, strides=4, activation='relu'),
            Conv2D(32, kernel_size=4, strides=2, activation='relu'),
            Flatten(),
            lambda conv_output: concatenate([conv_output, labeled_data_input]),
            Dense(256, activation='relu'),
            Dense(num_actions, activation='relu')
        ],
        aux_inputs=[labeled_data_input])

    model.compile('adam', loss='mse')

    if weights_file_path:
        if not os.path.exists('data'):
            os.makedirs('data')

        if os.path.isfile(weights_file_path):
            model.load_weights(weights_file_path)

    return model

def connect_layers(layers, aux_inputs=None, aux_outputs=None):
    if aux_inputs is None:
        aux_inputs = []

    if aux_outputs is None:
        aux_outputs = []

    layer_acc = layers[0]
    for layer in layers[1:]:
        layer_acc = layer(layer_acc)

    return Model(inputs=[layers[0]] + aux_inputs, outputs=[layer_acc] + aux_outputs)

