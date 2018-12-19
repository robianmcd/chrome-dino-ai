import time
import os

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, Flatten
from pynput.keyboard import Key, Controller

from src.experience_replay import ExperienceReplay
from src.game_environment import GameEnvironment
from src.dino_img_util import DinoImageUtil

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

keyboard = Controller()
img_util = DinoImageUtil()
weights_file_path = 'data/model_weights.h5'
monitor_id = 1

games_per_batch = 200
min_random_action_chance = 0.1
delay_random_actions = 0 # don't start random actions until 75% of the last batch's average duration
frames_before_min_random_action = 100000
ideal_fps = 6
ideal_frame_length = 1 / ideal_fps

train_state = {
    'random_action_chance': 1,
    'last_batch_avg_duration': 0
}


game = GameEnvironment(monitor_id)

#Architecture based on Deep Mind https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
model = Sequential()
model.add(Conv2D(16, kernel_size=8, strides=4, input_shape=(38,150,1), activation='relu'))
model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(game.actions), activation='relu'))
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

            screenshot_prev = None
            screenshot = None
            action_i_prev = None
            action_i = None
            game_over = False

            game_start_time = time.time()

            while not game_over:
                frame_start_time = time.time()
                action_i_prev = action_i
                action_i = None
                screenshot_prev = screenshot

                # ************ Observe current game state ************
                (screenshot, reward_instantaneous, game_over) = game.observe()

                # ************ Preform next action ************
                if not game_over:
                    action_i = pick_next_action(game_start_time, screenshot)
                    game.actions[action_i]['doAction']()

                # ************ Save to memory ************
                if screenshot_prev is not None:
                    exp_replay.remember({
                        'inputBefore': screenshot_prev,
                        'actionI': action_i_prev,
                        'rewardInstantaneous': reward_instantaneous,
                        'gameOver': game_over,
                        'inputAfter': screenshot
                    })

                # ************ Wait for frame to end ************
                if not game_over:
                    sleep_rest_of_frame(frame_start_time)

            total_batch_duration += time.time() - game_start_time

            game.reset_inputs()

            inputs, targets = exp_replay.get_batch(batch_size=64)
            model.train_on_batch(inputs, targets)

        model.save_weights(weights_file_path)
        exp_replay.save_memory()

        train_state['last_batch_avg_duration'] = total_batch_duration / games_per_batch
        print(f'Average time alive: {round(train_state["last_batch_avg_duration"], 2)}s. Random factor: {round(train_state["random_action_chance"], 3)}. Batch: {batch_i}.')

def pick_next_action(game_start_time, screenshot):
    # get next action
    if time.time() - game_start_time > train_state['last_batch_avg_duration'] * delay_random_actions and \
            np.random.rand() <= train_state['random_action_chance']:
        action_i = np.random.randint(0, len(game.actions), size=1)[0]
    else:
        #Array containing the predicted cumulative reward for each action
        action_reward_cumulative = model.predict(np.array([screenshot]))[0]
        action_i: int = np.argmax(action_reward_cumulative)

    if train_state['random_action_chance'] > min_random_action_chance:
        train_state['random_action_chance'] -= 1 / frames_before_min_random_action
        train_state['random_action_chance'] = max(train_state['random_action_chance'], min_random_action_chance)

    return action_i

def sleep_rest_of_frame(frame_start_time):
    frame_length = time.time() - frame_start_time
    if frame_length < ideal_frame_length:
        # print(f'sleeping for {ideal_frame_length - last_frame_length}s')
        time.sleep(ideal_frame_length - frame_length)
    else:
        # print(f'Last frame took too long: {last_frame_length}s')
        pass

#Code to profile
# with PyCallGraph(output=GraphvizOutput()):
#     train()

train()