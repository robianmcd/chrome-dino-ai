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
