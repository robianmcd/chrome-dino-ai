import time
import os

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, Flatten
from numpy.core.multiarray import ndarray
from pynput.keyboard import Key, Controller

from src.experience_replay import ExperienceReplay
from src.dino_img_util import DinoImageUtil

keyboard = Controller()
img_util = DinoImageUtil()
weights_file_path = 'data/model_weights.h5'
monitor_id = 1
monitor_dim = img_util.get_monitor_dimensions(monitor_id)
game_area = {
    'top': monitor_dim['top'] + 35,
    'left': int(monitor_dim['left'] + (monitor_dim['width'] / 2) - 300),
    'width': 600,
    'height': 150,
    'mon': monitor_id
}

games_per_batch = 200
random_action_chance = 1
min_random_action_chance = 0.15
delay_random_actions = 0.5 # don't start random actions until 50% of the last batch's average duration
frames_before_min_random_action = 100000
ideal_fps = 8
ideal_frame_length = 1 / ideal_fps

actions = [
    {'name': 'jumpKeyDown', 'doAction': lambda: keyboard.press(Key.up)},
    {'name': 'jumpKeyUp', 'doAction': lambda: keyboard.release(Key.up)},
    {'name': 'doNothing', 'doAction': lambda: ()},
    {'name': 'duckKeyDown', 'doAction': lambda: keyboard.press(Key.down)},
    {'name': 'duckKeyUp', 'doAction': lambda: keyboard.release(Key.down)}
]

#Architecture based on Deep Mind https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
model = Sequential()
model.add(Conv2D(16, kernel_size=8, strides=4, input_shape=(38,150,1), activation='relu'))
model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(actions), activation='relu'))
model.compile(optimizer='adam', loss='mse')

if not os.path.exists('data'):
    os.makedirs('data')

if os.path.isfile(weights_file_path):
    model.load_weights(weights_file_path)

exp_replay = ExperienceReplay(model=model, max_memory=100000, discount=.9)
exp_replay.load_memory()

batch_i = 0
last_batch_avg_duration = 0
while 42:
    total_batch_duration = 0
    batch_i += 1

    #for some reason the character seems to move to the right over time so this resets it
    keyboard.press(Key.f5)
    keyboard.release(Key.f5)
    time.sleep(2)

    for i in range(games_per_batch):
        #start game
        time.sleep(1)
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        time.sleep(3)

        input_prev = None
        input = img_util.get__processed_screenshot_np(game_area)
        game_over = False

        last_tick_time = time.time()
        game_start_time = time.time()

        while not game_over:

            this_tick_time = time.time()
            last_frame_length = this_tick_time - last_tick_time
            if last_frame_length < ideal_frame_length:
                time.sleep(ideal_frame_length - last_frame_length)
            else:
                #print(f'Last frame took too long: {last_frame_length}s')
                pass
            last_tick_time = this_tick_time

            input_prev = input

            # get next action
            if time.time() - game_start_time > last_batch_avg_duration * delay_random_actions and \
                    np.random.rand() <= random_action_chance:
                action_i = np.random.randint(0, len(actions), size=1)[0]
                #print(f'random: {actions[action_i]["name"]}')
            else:
                #Array containing the predicted cumulative reward for each action
                action_reward_cumulative = model.predict(np.array([input_prev]))[0]
                action_i: int = np.argmax(action_reward_cumulative)
                #print(actions[action_i]["name"])

            actions[action_i]['doAction']()
            input = img_util.get__processed_screenshot_np(game_area)

            #if the screen has stopped changing then the game must be over
            if(np.array_equal(input_prev, input)):
                game_over = True
                reward_instantaneous = 0
                #print('Dead')
            else:
                reward_instantaneous = 1

            exp_replay.remember({
                'inputBefore': input_prev,
                'actionI': action_i,
                'rewardInstantaneous': reward_instantaneous,
                'gameOver': game_over,
                'inputAfter': input
            })

            if random_action_chance > min_random_action_chance:
                random_action_chance -= 1 / frames_before_min_random_action
                random_action_chance = max(random_action_chance, min_random_action_chance)

        total_batch_duration += time.time() - game_start_time

        keyboard.release(Key.up)
        keyboard.release(Key.down)

        inputs, targets = exp_replay.get_batch(batch_size=32)
        model.train_on_batch(inputs, targets)

    model.save_weights(weights_file_path)
    exp_replay.save_memory()

    last_batch_avg_duration = total_batch_duration / games_per_batch
    print(f'Average time alive: {round(last_batch_avg_duration, 2)}s. Random factor: {round(random_action_chance, 3)}. Batch: {batch_i}.')