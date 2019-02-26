import os

from keras.engine.topology import InputLayer
from keras.models import Model
from keras.layers import Conv2D
from flask import Flask, jsonify, send_from_directory

from src.game_environment import GameEnvironment
from src.state_machine_game_environment import StateMachineGameEnvironment
import src.model_util as model_util
from src.experience_replay import ExperienceReplay
import kmri

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

exp_replay = ExperienceReplay(model=model, max_memory=200000, discount=.9)
exp_replay.load_memory()

img_inputs, labeled_inputs, targets = exp_replay.get_short_term_batch(num_frames_before_death=50, num_deaths=20)
kmri.visualize_model(model, [img_inputs, labeled_inputs])