import os

from keras.engine.topology import InputLayer
from keras.models import Model
from keras.layers import Conv2D
from flask import Flask, jsonify, send_from_directory

from src.game_environment import GameEnvironment
from src.state_machine_game_environment import StateMachineGameEnvironment
import src.model_util as model_util
from src.experience_replay import ExperienceReplay

game = GameEnvironment()
# game = StateMachineGameEnvironment()

model = model_util.get_model(
    img_width=150,
    img_height=38,
    num_labeled_inputs=game.get_num_labeled_inputs(),
    num_actions=len(game.actions),
    weights_file_path='data/model_weights.h5'
)

layer_metadata = {
    'dense_2': {
        'max': 'auto'
    }
}

layer_names = [layer.name for layer in model.layers]

layer_outputs = [layer.output for layer in model.layers if not isinstance(layer, InputLayer)]
wrappedModel = Model(inputs=model.inputs, outputs=layer_outputs)

exp_replay = ExperienceReplay(model=model, max_memory=200000, discount=.9)
exp_replay.load_memory()

static_folder = 'network_visualizer_ui/'
static_folder_relative_to_cwd = os.path.dirname(__file__) + '/' + static_folder

state = {
    'batch_i': 0,
    'batch': ([],[],[])
}

def load_batch():
    state['batch_i'] = 0
    state['batch'] = exp_replay.get_short_term_batch(num_frames_before_death=50, num_deaths=20)

app = Flask(__name__, static_folder=static_folder)

# Serve UI
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(static_folder_relative_to_cwd + path):
        return send_from_directory(static_folder, path)
    else:
        return send_from_directory(static_folder, 'index.html')

@app.route("/model")
def get_model():
    model_config = model.get_config()

    for layer_config in model_config['layers']:
        layer = model.get_layer(layer_config['name'])
        #Need to convert everything to int as it can be np.int32 which is not json serializable
        layer_config['outputShape'] = [int(x) for x in layer.output_shape[1:]]

    return jsonify(model_config)

@app.route("/predict")
def predict():
    if state['batch_i'] >= len(state['batch'][0]):
        load_batch()

    (img_inputs, labeled_inputs, targets) = state['batch']

    i = state['batch_i']
    inputs = [img_inputs[i:i+1], labeled_inputs[i:i+1]]

    state['batch_i'] += 1

    outputs = wrappedModel.predict(inputs)

    layer_outputs = {}

    for layer in model.layers:
        if isinstance(layer, InputLayer):
            layer_outputs[layer.name] = inputs.pop(0).tolist()[0]
        else:
            layer_outputs[layer.name] = outputs.pop(0).tolist()[0]

    return jsonify(layer_outputs)

app.run()
