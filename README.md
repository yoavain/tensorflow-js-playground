![visitors](https://visitor-badge.glitch.me/badge?page_id=yoavain.tf-playground)
[![CodeQL](https://github.com/yoavain/tensorflow-js-playground/workflows/CodeQL/badge.svg)](https://github.com/yoavain/tensorflow-js-playground/actions?query=workflow%3ACodeQL)

#### CUDA Environment Setup

```
Install CUDA 10.0
Download cUdNN v7.x for CUDA 10.0 for Windows 10
Extract cuDNN to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\
```

#### Conda Environment Setup
```
conda create --name tf_gpu tensorflow-gpu 
```

#### Conda Environment Activate
```
activate tf_gpu
```

#### NodeJS Setup
```
npm install
```

#### Run Demo
```
npm run image-annotate-demo
```