from keras.models import Model


def connect_layers(layers, aux_inputs=None, aux_outputs=None):
    if aux_inputs is None:
        aux_inputs = []

    if aux_outputs is None:
        aux_outputs = []

    layer_acc = layers[0]
    for layer in layers[1:]:
        layer_acc = layer(layer_acc)

    return Model(inputs=[layers[0]] + aux_inputs, outputs=[layer_acc] + aux_outputs)