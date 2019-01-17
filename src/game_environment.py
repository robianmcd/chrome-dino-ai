import time
import numpy as np

from pynput.keyboard import Key, Controller
from src.dino_img_util import DinoImageUtil
from src.window_mgr import ChromeWindowMgr

keyboard = Controller()
img_util = DinoImageUtil()
window_mgr = ChromeWindowMgr()

class GameEnvironment():
    def __init__(self, monitor_id=1):
        self.actions = [
            {'name': 'jumpKeyDown', 'doAction': lambda: self._trigger_key(Key.up, True)},
            {'name': 'jumpKeyUp', 'doAction': lambda: self._trigger_key(Key.up, False)},
            {'name': 'doNothing', 'doAction': lambda: ()},
            {'name': 'duckKeyDown', 'doAction': lambda: self._trigger_key(Key.down, True)},
            {'name': 'duckKeyUp', 'doAction': lambda: self._trigger_key(Key.down, False)}
        ]

        monitor_dim = img_util.get_monitor_dimensions(monitor_id)
        self.game_area = {
            'top': monitor_dim['top'] + 35,
            'left': int(monitor_dim['left'] + (monitor_dim['width'] / 2) - 300),
            'width': 600,
            'height': 150,
            'mon': monitor_id
        }

        self.key_states = {
            Key.up: False,
            Key.down: False
        }

        self.last_screenshot = None
        self.game_start_time = None

    def _trigger_key(self, key, press):
        if(press):
            keyboard.press(key)
        else:
            keyboard.release(key)

        self.key_states[key] = press

    #for some reason the character seems to move to the right over time if the game isn't reloaded
    def reload(self):
        window_mgr.set_foreground()
        keyboard.press(Key.f5)
        keyboard.release(Key.f5)
        self.last_screenshot = None
        self.game_start_time = None

    def start_and_sleep_until_ready(self):
        time.sleep(1)
        self._trigger_key(Key.up, True)
        self._trigger_key(Key.up, False)
        time.sleep(3)
        self.game_start_time = time.time()

    def reset_controller_inputs(self):
        self._trigger_key(Key.up, False)
        self._trigger_key(Key.down, False)

    def observe(self):
        screenshot = img_util.get_processed_screenshot_np(self.game_area)

        #If a game takes over an hour then normalized_time maxes out at 1
        normalized_time = min((time.time() - self.game_start_time) / (60 * 60), 1)
        labeled_data = np.array([self.key_states[Key.up], self.key_states[Key.down], normalized_time])

        #if the screen has stopped changing then the game must be over
        if np.array_equal(self.last_screenshot, screenshot):
            #TODO: figure out why this is necessary or just come up with a less hacky way to check if the game is over
            screenshot = img_util.get_processed_screenshot_np(self.game_area)
            if np.array_equal(self.last_screenshot, screenshot):
                game_over = True
                reward_instantaneous = 0
            else:
                game_over = False
                reward_instantaneous = 1
                print('Duplicate Screenshots')

        else:
            game_over = False
            reward_instantaneous = 1

        if game_over and time.time() - self.game_start_time < 0.3:
            img_util.save_screenshot(self.last_screenshot, 'data/before_immediate_death.png')
            img_util.save_screenshot(screenshot, 'data/after_immediate_death.png')
            raise Exception('Died immediately after start')

        self.last_screenshot = screenshot

        return (screenshot, labeled_data, reward_instantaneous, game_over)

    def get_num_labeled_inputs(self):
        return 3