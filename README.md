# rl-transfer-learning

## installation

### create virtual environment

`python3 -m venv RLenv` 

`source RLenv/bin/activate` 

### install MuJoCo

clone https://github.com/openai/mujoco-py/

in the cloned directory, with RLenv, run:

`pip3 install -r requirements.txt`

`pip3 install -r requirements.dev.txt`

`python3 setup.py install`

### install Meta-World

`pip3 install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld`

test if Meta-World is installed correctly, in a python console run: `import metaworld`

if the above command fails, install some additional libraries that may be required (ubuntu):

`sudo apt-get install libosmesa6-dev`

```
sudo add-apt-repository ppa:jamesh/snap-support 
sudo apt-get update
sudo apt install patchelf
```


### install other requirements

in the project root directory run:

`pip3 install -r requirements.txt`

