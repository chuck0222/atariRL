# AtariRL
For dependencies, do: pip install -r requirements.txt while in the DQN folder.
To run the AtariDQN.py file, go into the DQN folder and run: python3 AtariDQN.py

Arguments to add for specific needs:
Add these arguments after AtariDQN.py if to be used:
  --env-id "Environment_Name" - This is for specifying specific environments, default game being Assault-v5
  --save-model True - This is whether to save given model being run, default is set to False
  --total-timesteps Number - This is how many steps the model should train, default set to 500, 000 steps
  --capture-video True - This is whether to capture a video while training, default is set to False
  --load "Saved_Model" - This is whether or not to load a model saved and run already, default is an empty String

CleanRL citation:
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}