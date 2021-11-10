# rl-transfer-learning

## installation on Euler cluster

`env2lmod`

`module load gcc/8.2.0`

`module load mesa/18.3.6`

`module load python_gpu/3.8.5`

### create virtual environment

`python3 -m venv RLenv` 

`source RLenv/bin/activate` 

### install MuJoCo
`cd $HOME`

`nano .bashrc`

go to the end of file and paste `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin` 

press `Ctrl+O` then `Enter` then `Ctrl+X`

`exec bash`

`curl -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz`

`mkdir .mujoco`

`tar -xvzf mujoco210-linux-x86_64.tar.gz -C $HOME/.mujoco/`

`mkdir MuJoCo`

`cd MuJoCo`

`git clone https://github.com/openai/mujoco-py/`

`cd mujoco-py`

(in the cloned directory) with RLenv active, run:

`pip3 install -r requirements.txt`

`pip3 install -r requirements.dev.txt`

`python3 setup.py install --user`

### install Meta-World

`pip3 install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld`

test if Meta-World is installed correctly in a python console:

`python3`

`import metaworld`

`print(metaworld.ML1.ENV_NAMES)`

if the above command gives errors, install some additional libraries that may be required (ubuntu):

`sudo apt-get install libosmesa6-dev`

```
sudo add-apt-repository ppa:jamesh/snap-support 
sudo apt-get update
sudo apt install patchelf
```


### install other requirements

in the project root directory run:

`pip3 install -r requirements.txt`

