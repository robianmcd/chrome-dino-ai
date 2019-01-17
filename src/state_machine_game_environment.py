#              State Machine
# Path for optimal reward is A -> B -> C -> A -> ...
#
#    +------+--------------+--------------+
#    |      |              |              |
#    |      |              |              |
# +--v--+   |   +------+   |   +------+   |
# |   +0+---+   |    +2+---+   |    +9+---+
# | A   |       | B    |       | C    |
# |   -1+------->    +0+------->    -1+---+
# +-----+       +------+       +---^--+   |
#                                  |      |
#                                  |      |
#                                  +------+

import os
import random
import numpy as np

import imageio

from src.dino_img_util import DinoImageUtil

img_util = DinoImageUtil()
script_path = os.path.dirname(os.path.realpath(__file__))

class StateMachineGameEnvironment():
    def __init__(self):
        self.actions = [
            {'name': 'forward', 'doAction': lambda: self._move('forward')},
            {'name': 'back', 'doAction': lambda: self._move('back')}
        ]

        self.state_machine = None
        self.step = None
        self.max_steps = 100
        self.last_reward = None

    def _move(self, direction):
        self.last_reward = self.state_machine.move(direction)
        self.step += 1

    def reload(self):
        pass

    def start_and_sleep_until_ready(self):
        self.state_machine = StateMachine()
        self.step = 0
        self.last_reward = 0

    def reset_controller_inputs(self):
        pass

    def observe(self):
        state = self.state_machine.current_state
        img_folder = self.state_machine.states[state]['img_folder']
        img_name = random.choice(os.listdir(img_folder))

        screenshot = imageio.imread(os.path.join(img_folder, img_name)) / 255
        screenshot = screenshot.reshape((38,150,1))

        labeled_data = np.array([self.step / self.max_steps])

        game_over = (self.step == self.max_steps)

        return (screenshot, labeled_data, self.last_reward, game_over)

    def get_num_labeled_inputs(self):
        return 1

class StateMachine():
    def __init__(self):
        self.states = {
            'A': {
                'forward': {'reward': -1, 'nextState': 'B'},
                'back': {'reward': 0, 'nextState': 'A'},
                'img_folder': os.path.join(script_path, '..', 'state_machine_images', 'ground')
            },
            'B': {
                'forward': {'reward': 0, 'nextState': 'C'},
                'back': {'reward': 2, 'nextState': 'A'},
                'img_folder': os.path.join(script_path, '..', 'state_machine_images', 'air')
            },
            'C': {
                'forward': {'reward': -1, 'nextState': 'C'},
                'back': {'reward': 9, 'nextState': 'A'},
                'img_folder': os.path.join(script_path, '..', 'state_machine_images', 'game_over')
            }
        }

        self.current_state = 'A'

    def move(self, direction):
        reward = self.states[self.current_state][direction]['reward']
        self.current_state = self.states[self.current_state][direction]['nextState']
        return reward