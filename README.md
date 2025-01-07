### SVD-with-ReferenceNet config

data prepare
```
cd assets/
mkdir data/

```

model weight
```
mkdir models
### SVD-xt-1-1
链接: https://pan.baidu.com/s/1qEF0qjQIhfp-pCBbFtI2Pw?pwd=qehm 
提取码: qehm 
### DWPose
mkdir DWPose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
### stable-diffusion-2-1
apt-get install git-lfs
cd assets/models
git lfs install 
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
```

environment config
```
conda create -n mimo python==3.10
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
apt install libgl1-mesa-glx
pip install -r requirements.txt
```

train
```
bash run_train_pose_animation.sh
```