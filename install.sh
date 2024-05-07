## install NVIDIA drivers
sudo apt install gcc make -y
wget https://ru.download.nvidia.com/XFree86/Linux-x86_64/550.54.14/NVIDIA-Linux-x86_64-550.54.14.run
sudo bash NVIDIA-Linux-x86_64-550.54.14.run

## Install Anaconda (optional)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

## 
git clone git@github.com:shigabeev/vits2-inference.git
cd vits2-inference/
wget https://huggingface.co/frappuccino/vits2_ru_natasha/resolve/main/natasha.onnx?download=true
pip install requirements.txt

mkdir model_repository
mkdir model_repository/natasha
mkdir model_repository/shergin
mv natasha.onnx\?download\=true model_repository/natasha/model.onnx

## create services
sudo cp services/* /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl start vits2_st.service
sudo systemctl start vits2_api.service
