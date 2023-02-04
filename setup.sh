# Install and set up pyenv
[ ! -d "$HOME/.pyenv" ] && curl https://pyenv.run | bash

# export to current vars
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# export to config
if [ -f "$HOME/.bashrc" ]
then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc 
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
elif [ -f "$HOME/.zshrc" ]
then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc 
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc 
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
fi

# Setup and install pyenv and install pipenv and pipenv-shebang
[ ! -d "$HOME/.pyenv/versions/3.8.2" ] && pyenv install 3.8.2
pyenv global 3.8.2
python3 -m pip install --user pipenv pipenv-shebang

# Configure bash or zsh 
export PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"
export PATH="$PATH:$PYTHON_BIN_PATH"

# export to config
if [ -f "$HOME/.bashrc" ]
then
    echo 'export PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"' >> ~/.bashrc
    echo 'export PATH="$PATH:$PYTHON_BIN_PATH"' >> ~/.bashrc
elif [ -f "$HOME/.zshrc" ] 
then
    echo 'export PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"'  >> ~/.zshrc
    echo 'export PATH="$PATH:$PYTHON_BIN_PATH"' >> ~/.zshrc 
fi

# To install dependencies to install and setup tensorfloq
brew install hdf5
export HDF5_DIR="$(brew --prefix hdf5)"
export PIP_NO_BINARY=h5py && pipenv install h5py

# install everything required
python3 -m pipenv install
