<div align="center">

# Simple RVC

A simple, high-quality voice conversion tool focused on simplicity and ease of use.


</div>


# Key Feature

* Pitch extraction methods like: pm, dio, crepe, fcpe, rmvpe, harvest, yin, pyin and more!


## INSTALL

**requires Python 3.10.x or 3.11.x and FFmpeg installed**

```
git clone https://github.com/SawitProject/Simple-RVC.git
```

NVIDIA:
```
python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```
AMD:
```
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision
python -m pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp311-none-win_amd64.whl
python -m pip install -r requirements.txt
```
CPU:
```
python -m pip install -r requirements.txt
```

## Run GUI

use terminal:
```
python gui.py
```
or run with window file:
```
open-gui.bat
```

## Model

**Click `Import Model (.zip)` and upload the zip model containing the .pth and .index weight files (If any)**

**Or load the model manually**

```
assets/models
├── Model 1
│   ├── model_name.pth
│   └── model_name.index
└── Model 2
    ├── model_name.pth
    └── model_name.index
```

## Disclaimer
- The RVC project is developed for research, educational, and personal entertainment purposes. I do not encourage, nor do I take any responsibility for, any misuse of voice conversion technology for fraudulent purposes, identity impersonation, or violations of privacy or copyright belonging to any individual or organization.

- Users are solely responsible for how they use this software and must comply with the laws and regulations of the country in which they reside or operate.

- The use of voices of celebrities, real people, or public figures must be authorized or ensured not to violate any applicable laws, ethical standards, or the rights of the individuals involved.

- The author of this project holds no legal liability for any consequences arising from the use of this software.

## Terms of Use
- You must ensure that any audio content you upload and convert through this project does not infringe upon the intellectual property rights of any third party.

- This project must not be used for any illegal activity, including but not limited to fraud, harassment, or causing harm to others.

- You are fully responsible for any damages resulting from improper use of the product.

- I am not liable for any direct or indirect damages arising from the use of this project.

# Planed Feature

* Music Separation (Full Model)


* Converting the RVC model to the ONNX model

* Embedded extraction supports models for: onnx (.onnx), transformers (.bin - .json), spin (.bin - .json), whisper (.pt).

* Gradio GUI

  
## Project-based construction


- **Algorithm: [Vietnamese RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)**
