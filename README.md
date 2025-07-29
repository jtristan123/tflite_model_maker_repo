# CCC Cone Detection with Coral Edge TPU

This project demonstrates how to train, convert, and deploy a custom object detection model (cone detector) on the Google Coral USB Accelerator using TensorFlow Lite and Edge TPU.

### 📦 What This Project Does

- Trains a MobileNet SSD model to detect cones using custom images.
- Converts and compiles the model for Coral Edge TPU.
- Runs live object detection using Raspberry Pi and a USB webcam with Picamera2.
- Sends control commands to a robot via serial based on object detection.

---

#### 🛠️ Environment Setup

### 🖥️ Training (on local PC using WSL + VSCode)
1. Clone the repo:
   ```
   git clone https://github.com/yourusername/ccc-coral-cone-detection
   cd ccc-coral-cone-detection
2. Create a Python 3.9 virtual environment (Coral TPU compiler is not compatible with 3.11):

```
sudo apt install python3.9 python3.9-venv python3.9-dev
python3.9 -m venv coral-env
source coral-env/bin/activate
```
3. Install training dependencies:

```
pip install -r requirements.txt
```
4. Train your model:

```
python3 train.py
```
Training uses tflite_model_maker and image-label pairs in images/ folder.<br>
### 📂 Folder Structure
```
├── images/                   # Training images (jpg) and labels (xml)
├── exported-model-v/         # Exported and compiled TFLite model goes here from train.py, this can all be moved to raspberry 
   ├── model.tflite           # .tflite model from train.py before compiler
   ├── labels.txt             # dont forget the labels.txt file  
├── train.py                  # Training and exports the model .tflite script
├── verify_if_int8.py         # verify the model is in int8 compabily with coral TFlite 


```

### 📦 Model Conversion
After training, verify your model is TensorFlow Lite:

```
python3 verify_if_int8.py
```
Then compile it for Edge TPU (on a Linux PC or in WSL):

```
edgetpu_compiler model.tflite
```
Output: model_edgetpu.tflite
move this file to the same folder as your model.tflite

### GPU VS CPU 

| Component | Model                       | Release Year | Specs                                                |
| --------- | --------------------------- | ------------ | ---------------------------------------------------- |
| **CPU**   | AMD Ryzen 7 3700X           | 2019         | 8 cores / 16 threads @ 3.6 GHz (boost up to 4.4 GHz) |
| **GPU**   | NVIDIA GeForce GTX 1060 6GB | 2016         | 1280 CUDA cores, 6 GB GDDR5                          |
| **RAM**   | DDR4 Memory                 | —            | 32 GB Total                                          |


#### ⏱️ Training Time Comparison
Device	Training Duration (100 epochs)

| Component         | Time            |
| ----------------- | --------------- |
|GPU (GTX 1060)     |	490.83 seconds|
|CPU (Ryzen 7 3700X)|	965.90 seconds|




📊 Performance Summary
Speedup with GPU: ~1.97× faster

GPU acceleration provided nearly 2× performance improvement over the CPU in this training task.

Despite the GTX 1060 being older, it still significantly boosts training due to TensorFlow’s GPU optimization.

#### Software specifications

CUDA Version	11.2<br>
cuDNN Version	8.1.1<br>
TensorFlow Version	2.8.4<br>
Python Version	3.9.x<br>
OS	Ubuntu 24.04 LTS (via WSL2)<br>

links used to get it to work
https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=runfilelocal
https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#download-nvidia-driver-and-cuda-software
downgrade GCC to 9.x 
cuDNN 8.1.1 from link need both this and cuda to use gpu with tensorflow 2.8

