# AtariRL
*Code adapted from CleanRL library*

First off, make sure the Python interpreter you are using is < 3.10, preferably version 3.9
To get needed dependencies, do: pip install gym torch wandb huggingface_hub.
To run on Windows, do: python3 .\AtariDQN.py --save-model True --track True --capture-video True
To track the model, you will need to create a Wandb account, and use the token in order to log performance for tracking. 
