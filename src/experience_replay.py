import numpy as np
import pickle
import os

class ExperienceReplay():
    def __init__(self, model, max_memory=10000, discount=.9):
        self.model = model
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, memory_unit):
        # memory[i] = {'inputBefore', 'actionI', 'rewardInstantaneous', 'gameOver', 'inputAfter'}
        memory_unit = memory_unit.copy()
        memory_unit['inputBefore'] = (memory_unit['inputBefore'] * 256).astype(np.uint8)
        memory_unit['inputAfter'] = (memory_unit['inputAfter'] * 256).astype(np.uint8)

        #if memory_unit['inputBefore'] is already in the memory then store a reference to it instead of duplicating it
        if self.memory and np.array_equal(self.memory[-1]['inputAfter'], memory_unit['inputBefore']):
            memory_unit['inputBefore'] = self.memory[-1]['inputAfter']

        self.memory.append(memory_unit)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def recall(self, i):
        memory_unit = self.memory[i].copy()
        memory_unit['inputBefore'] = memory_unit['inputBefore'] / 256
        memory_unit['inputAfter'] = memory_unit['inputAfter'] / 256

    def get_batch(self, batch_size=10):
        memory_size = len(self.memory)
        batch_size = min(batch_size, memory_size)
        num_actions = self.model.output_shape[-1]

        inputs = np.zeros((batch_size, self.model.input_shape[1], self.model.input_shape[2], 1))
        targets = np.zeros((batch_size, num_actions))
        for i, memory_i in enumerate(np.random.randint(0, memory_size, size=batch_size)):
            memory_unit = self.memory[memory_i]
            game_over = memory_unit['gameOver']

            inputs[i] = memory_unit['inputBefore']
            targets[i] = self.model.predict(np.array([memory_unit['inputBefore']]))[0]

            #The expected cumulative reward you will get from the rest of the actions
            reward_remaining_cumulative = np.max(self.model.predict(np.array([memory_unit['inputAfter']]))[0])
            if game_over:
                targets[i, memory_unit['actionI']] = memory_unit['rewardInstantaneous']
            else:
                targets[i, memory_unit['actionI']] = memory_unit['rewardInstantaneous'] + self.discount * reward_remaining_cumulative
        return inputs, targets

    def save_memory(self, data_folder='data'):
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        with open(os.path.join(data_folder, 'memory.pickle'), 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)

    def load_memory(self, data_folder='data'):
        file_path = os.path.join(data_folder, 'memory.pickle')
        if os.path.exists(file_path):
            with open(file_path,'rb') as memory_file:
                self.memory = pickle.load(memory_file)