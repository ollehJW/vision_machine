# Vision Machine

Image Classification을 수행할 수 있는 Code



# 사용법
1. requirements.txt 파일에 있는 패키지를 설치한다. (pip install -r requirements.txt)
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch  
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch  
- Transformer 사용 시 timm 패키지 설치 필요  
pip install timm



2. vision_machine_parameters.params을 목적에 맞게 수정한다.

(사용 가능 모델: vgg16, resnet50, mobilenet_v2, vision transformer, )

3. vision_machine.py 를 실행한다.