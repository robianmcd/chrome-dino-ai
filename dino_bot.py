import time
import os

import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate

from src.experience_replay import ExperienceReplay
from src.game_environment import GameEnvironment
from src.dino_img_util import DinoImageUtil
from src.keras_util import connect_layers

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

img_util = DinoImageUtil()
weights_file_path = 'data/model_weights.h5'
monitor_id = 1

games_per_batch = 200
min_random_action_chance = 0.1
delay_random_actions = 0 # don't start random actions until 75% of the last batch's average duration
frames_before_min_random_action = 200000
ideal_fps = 8
ideal_frame_length = 1 / ideal_fps

train_state = {
    'random_action_chance': 1,
    'last_batch_avg_duration': 0
}


game = GameEnvironment(monitor_id)

#Architecture based on Deep Mind https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
labeled_data_input = Input(shape=(3,), name='labeledInput')
model = connect_layers(
    [
        Input(shape=(38,150,1), name='imgInput'),
        Conv2D(16, kernel_size=8, strides=4, activation='relu'),
        Conv2D(32, kernel_size=4, strides=2, activation='relu'),
        Flatten(),
        lambda conv_output: concatenate([conv_output, labeled_data_input]),
        Dense(256, activation='relu'),
        Dense(len(game.actions), activation='relu')
    ],
    aux_inputs=[labeled_data_input])

model.compile('adam', loss='mse')

if not os.path.exists('data'):
    os.makedirs('data')

if os.path.isfile(weights_file_path):
    model.load_weights(weights_file_path)

exp_replay = ExperienceReplay(model=model, max_memory=200000, discount=.9)
exp_replay.load_memory()

def train():
    batch_i = 0
    while(42):
        total_batch_duration = 0
        batch_i += 1

        for i in range(games_per_batch):
            game.reload()
            game.start_and_sleep_until_ready()

            img_data_prev = None
            img_data = None
            labeled_data_prev = None
            labeled_data = None
            action_i_prev = None
            action_i = None
            game_over = False

            game_start_time = time.time()

            while not game_over:
                frame_start_time = time.time()
                action_i_prev = action_i
                action_i = None
                img_data_prev = img_data
                labeled_data_prev = labeled_data

                # ************ Observe current game state ************
                (img_data, labeled_data, reward_instantaneous, game_over) = game.observe()

                # ************ Preform next action ************
                if not game_over:
                    action_i = pick_next_action(game_start_time, img_data, labeled_data)
                    game.actions[action_i]['doAction']()

                # ************ Save to memory ************
                if img_data_prev is not None:
                    exp_replay.remember({
                        'imgDataBefore': img_data_prev,
                        'labeledDataBefore': labeled_data_prev,
                        'actionI': action_i_prev,
                        'rewardInstantaneous': reward_instantaneous,
                        'gameOver': game_over,
                        'imgDataAfter': img_data,
                        'labeledDataAfter': labeled_data
                    })

                # ************ Wait for frame to end ************
                if not game_over:
                    sleep_rest_of_frame(frame_start_time)

            total_batch_duration += time.time() - game_start_time

            game.reset_controller_inputs()

            (img_inputs, labeled_inputs, targets) = exp_replay.get_batch(batch_size=64)
            model.train_on_batch([img_inputs, labeled_inputs], targets)

        model.save_weights(weights_file_path)
        exp_replay.save_memory()

        train_state['last_batch_avg_duration'] = total_batch_duration / games_per_batch
        print(f'Average time alive: {round(train_state["last_batch_avg_duration"], 2)}s. Random factor: {round(train_state["random_action_chance"], 3)}. Batch: {batch_i}.')

def pick_next_action(game_start_time, img_data, labeled_data):
    # get next action
    if time.time() - game_start_time > train_state['last_batch_avg_duration'] * delay_random_actions and \
            np.random.rand() <= train_state['random_action_chance']:
        action_i = np.random.randint(0, len(game.actions), size=1)[0]
    else:
        #model.predict expects an array of samples but we only want it to predict one so we need to wrap inputs in arrays
        img_input = np.array([img_data])
        labeled_input = np.array([labeled_data])

        action_reward_cumulative = model.predict([img_input, labeled_input])[0]
        action_i: int = np.argmax(action_reward_cumulative)

    if train_state['random_action_chance'] > min_random_action_chance:
        train_state['random_action_chance'] -= 1 / frames_before_min_random_action
        train_state['random_action_chance'] = max(train_state['random_action_chance'], min_random_action_chance)

    return action_i

def sleep_rest_of_frame(frame_start_time):
    frame_length = time.time() - frame_start_time
    if frame_length < ideal_frame_length:
        # print(f'sleeping for {ideal_frame_length - frame_length}s')
        time.sleep(ideal_frame_length - frame_length)
    else:
        print(f'Last frame took too long: {frame_length}s')
        pass

# # Code to profile
# with PyCallGraph(output=GraphvizOutput()):
#     train()

train()