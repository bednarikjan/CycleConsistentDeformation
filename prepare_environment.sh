cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
/home/bednarik/miniconda/bin/conda init
source ~/.bashrc
cd /cvlabdata2/home/jan/projects/3rd_party/CycleConsistentDeformation/cyccon
conda env create -f environment_new.yml
conda activate cycleconsist
conda config --set auto_activate_base false
conda install -y pytorch torchvision -c pytorch
pip install h5py
pip install pyYAML
cd /cvlabdata2/home/jan/projects/3rd_party/CycleConsistentDeformation/cyccon/extension
python setup.py install
