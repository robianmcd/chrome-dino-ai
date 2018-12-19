import time
import numpy as np

from pynput.keyboard import Key, Controller
from src.dino_img_util import DinoImageUtil

keyboard = Controller()
img_util = DinoImageUtil()

class GameEnvironment():
    def __init__(self, monitor_id=1):
        self.actions = [
            {'name': 'jumpKeyDown', 'doAction': lambda: keyboard.press(Key.up)},
            {'name': 'jumpKeyUp', 'doAction': lambda: keyboard.release(Key.up)},
            {'name': 'doNothing', 'doAction': lambda: ()},
            {'name': 'duckKeyDown', 'doAction': lambda: keyboard.press(Key.down)},
            {'name': 'duckKeyUp', 'doAction': lambda: keyboard.release(Key.down)}
        ]

        monitor_dim = img_util.get_monitor_dimensions(monitor_id)
        self.game_area = {
            'top': monitor_dim['top'] + 35,
            'left': int(monitor_dim['left'] + (monitor_dim['width'] / 2) - 300),
            'width': 600,
            'height': 150,
            'mon': monitor_id
        }

        self.last_screenshot = None


    #for some reason the character seems to move to the right over time if the game isn't reloaded
    def reload(self):
        keyboard.press(Key.f5)
        keyboard.release(Key.f5)
        self.last_screenshot = None

    def start_and_sleep_until_ready(self):
        time.sleep(1)
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        time.sleep(3)

    def reset_inputs(self):
        keyboard.release(Key.up)
        keyboard.release(Key.down)

    def observe(self):
        screenshot = img_util.get__processed_screenshot_np(self.game_area)

        #if the screen has stopped changing then the game must be over
        if np.array_equal(self.last_screenshot, screenshot):
            #TODO: figure out why this is necessary or just come up with a less hacky way to check if the game is over
            screenshot = img_util.get__processed_screenshot_np(self.game_area)
            if np.array_equal(self.last_screenshot, screenshot):
                game_over = True
                reward_instantaneous = 0
                print('Dead')
            else:
                game_over = False
                reward_instantaneous = 1

        else:
            game_over = False
            reward_instantaneous = 1

        self.last_screenshot = screenshot

        return (screenshot, reward_instantaneous, game_over)
