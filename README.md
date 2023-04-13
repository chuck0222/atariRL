# AtariRL

First off, make sure the Python interpreter you are using is < 3.10, preferably version 3.9
To get needed dependencies, do: pip install gym torch wandb huggingface_hub.
To run on Windows, do: python3 .\AtariDQN.py --save-model True --track True --capture-video True
To track the model, you will need to create a Wandb account, and use the token in order to log performance for tracking. 

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