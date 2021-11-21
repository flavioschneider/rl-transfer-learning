import os

os.chdir('./rl-starter-files')
os.system('python scripts/train.py --algo ppo --env MiniGrid-FourRooms-4x4-v0 --save-interval 10 --frames 80000')

