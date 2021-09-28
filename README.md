
<!-- PROJECT TITLE -->
<h1 align="center">Face Mask Detection Using Deep Learning</h1>

<!-- PROJECT DESCRIPTION -->
## ➲ Project description
Face mask detection implementation using CNN model with keras and opencv

<!-- PREREQUISTIES -->
## ➲ Prerequisites
This is list of required packages and modules for the project to be installed :
* Python 3.x

  <https://www.python.org/downloads>
  
* OpenCV 
  ```sh
  pip install opencv-python
  ```
* Numpy 
  ```sh
  pip install numpy
  ```
* Tensorflow 
  ```sh
  pip install tensorflow
  ```
* Keras 
  ```sh
  pip install keras
  ```

<!-- INSTALLATION -->
## ➲ Installation
1. Clone the repo
   ```sh
   git clone https://github.com/omaarelsherif/Face-Mask-Detection-Using-Deep-Learning.git
   ```
2. Run the code from cmd
   ```sh
   python face_mask_detection.py "FLAG" "IMAGE_PATH"/"VIDEO_PATH"
   ```
   For Image:
   ```sh
   python face_mask_detection.py img Images/img1.jpg
   ```
   For Video:
   ```sh
   python face_mask_detection.py vid Videos/video.mp4
   ```
   For Cam:
   ```sh
   python face_mask_detection.py cam
   ```

<!-- OUTPUT -->
## ➲ Output
Here's the project output where the input is an image containing single or multi faces or a video and the output will be the same image with bounding boxs around all faces and a label showing if a person face wearing mask or not and same thing with video or cam as follows:

<h3>Face Mask Detection - Image Output</h3>

![alt text for screen readers](/Outputs/output.jpg "Face Mask Detection Image Output")

<h3>Face Mask Detection - Video Output</h3>

![alt text for screen readers](/Outputs/output.gif "Face Mask Detection Video Output")


<!-- REFERENCES -->
## ➲ References
These links may help you to better understanding of the project idea and techniques used :
1. CNN for image classification : https://bit.ly/3D6zCvx
2. OpenCV haarcascade : https://bit.ly/39HWt3D
   
<!-- CONTACT -->
## ➲ Contact
- E-mail   : [omaarelsherif@gmail.com](mailto:omaarelsherif@gmail.com)
- LinkedIn : https://www.linkedin.com/in/omaarelsherif/
- Facebook : https://www.facebook.com/omaarelshereif
  