import numpy as np
import pickle
import os

class ExperienceReplay():
    def __init__(self, model, max_memory=100000, discount=.9):
        self.model = model
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, memory_unit):
        # memory[i] = {'imgDataBefore', 'labeledDataBefore', 'actionI', 'rewardInstantaneous', 'gameOver', 'imgDataAfter', 'labeledDataAfter'}
        memory_unit = memory_unit.copy()
        memory_unit['imgDataBefore'] = (memory_unit['imgDataBefore'] * 256).astype(np.uint8)
        memory_unit['imgDataAfter'] = (memory_unit['imgDataAfter'] * 256).astype(np.uint8)

        #if memory_unit['imgDataBefore'] is already in the memory then store a reference to it instead of duplicating it
        if self.memory and np.array_equal(self.memory[-1]['imgDataAfter'], memory_unit['imgDataBefore']):
            memory_unit['imgDataBefore'] = self.memory[-1]['imgDataAfter']

        self.memory.append(memory_unit)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def recall(self, i):
        memory_unit = self.memory[i].copy()
        memory_unit['imgDataBefore'] = memory_unit['imgDataBefore'] / 256
        memory_unit['imgDataAfter'] = memory_unit['imgDataAfter'] / 256
        return memory_unit

    def get_batch(self, batch_size=32):
        memory_size = len(self.memory)
        batch_size = min(batch_size, memory_size)
        num_actions = self.model.output_shape[-1]

        #(batch_size=None, rows=38, columns=150, channels=1)
        (_, img_height, img_width, _) = self.model.get_layer('imgInput').input_shape
        img_inputs = np.zeros((batch_size, img_height, img_width, 1))
        labeled_inputs = np.zeros((batch_size, 3))
        targets = np.zeros((batch_size, num_actions))
        for i, memory_i in enumerate(np.random.randint(0, memory_size, size=batch_size)):
            memory_unit = self.recall(memory_i)
            game_over = memory_unit['gameOver']

            img_inputs[i] = memory_unit['imgDataBefore']
            labeled_inputs[i] = memory_unit['labeledDataBefore']

            #model.predict expects an array of samples but we only want it to predict one so we need to wrap inputs in arrays
            img_input_before = np.array([memory_unit['imgDataBefore']])
            labeled_input_before = np.array([memory_unit['labeledDataBefore']])
            targets[i] = self.model.predict([img_input_before, labeled_input_before])[0]

            if game_over:
                targets[i, memory_unit['actionI']] = memory_unit['rewardInstantaneous']
            else:
                img_input_after = np.array([memory_unit['imgDataAfter']])
                labeled_input_after = np.array([memory_unit['labeledDataAfter']])
                #The expected cumulative reward you will get from the rest of the actions
                reward_remaining_cumulative = np.max(self.model.predict([img_input_after, labeled_input_after])[0])

                targets[i, memory_unit['actionI']] = memory_unit['rewardInstantaneous'] + self.discount * reward_remaining_cumulative

        return img_inputs, labeled_inputs, targets

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