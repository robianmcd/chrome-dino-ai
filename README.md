 # Chrome Dino AI
 
AI bot that learns to play the offline dinosaur game in Chrome.

 # Install
 `pip install -r requirements.txt`
 
 # Run
 
 - open chrome dev tools and "undock in separate window"
 - Select the offline checkbox in the dev tools
 - try to go to a website and put chrome in full screen
 - run `python dino_bot.py`
 - click on the chrome window so the bot can send it keyboard input
 - wait a couple days
 - profit
 
 Note: if you have multiple monitors, uncomment the "show image" code in `src/dino_img_util.py` to see which screen the bot is looking at. You can control which monitor it uses by setting `monitor_id` in `dino_bot.py`
 
  # References
  - [Beat Atari with Deep Reinforcement Learning!](https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26)
  - [Keras plays catch](https://gist.github.com/EderSantana/c7222daa328f0e885093)

 # TODOs / Experiments
 
 - Pass keyboard sate and run duration into the model
 - use an RNN with stateful=True
   - https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
   - https://stackoverflow.com/questions/43882796/when-does-keras-reset-an-lstm-state
 - Increase dense layer to 256 neurons to match the atari blog
 - Try optimizer described in atari blog
    - `keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)`
 - Save/Load `random_action_chance` so you can kill the process and pickup where you left off
 - Add ability to train initial model on everything in the memory
 - Compress pixels to binary in the memory (need to use full size image instead of scaled down one so there is no anti-aliasing)
 - Focus on chrome window at the start of each run
 - Profile game loop. See how high the fps can go
 - Seem how many duplicate images are in the memory
 - Selectively prefer memories further from the start and ones close to the end of a run
 - Clean up code in dino_bot to make it easier to understand / more declarative
 - Unit test utility functions
 