Computational Vision - Studies on face detection and facial recognition with Python. This study uses the dlib, opencv and cvlib libraries to detect and classify faces by assigning a name.

-------------------------------------------------------------------------------------------------------------

Information about "visao_compu.yml " file:

visao_compu.yml -  Information about the packages and versions used in the programs.


-------------------------------------------------------------------------------------------------------------
Information about ".py" files:

face_recognition_cvlib_train.py – program responsible for the training to extract the main characteristics of the images. This program generates the files with the descriptors of the images.

face_recognition_cvlib_test.py – program responsible for evaluating the performance of the model trained by the "face_recognition_cvlib_train.py". This program provides some statistics about face recognition.

face_recognition_cvlib.py – after the training (face_recognition_cvlib_train.py) and testing (face_recognition_cvlib_test.py ) of the model, this program is responsible for batch processing to classified and save the image with the classification assigned to it.

face_recognition_webcam_cvlib.py - after the training (face_recognition_cvlib_train.py) and testing (face_recognition_cvlib_test.py ) of the model, this program is responsible to classify an image captured by the webcam.

myhelper.py - Miscellaneous support functions.

-------------------------------------------------------------------------------------------------------------

Directory Information:

resources – directory where the training files are saved.

dlib_resources – directory of files used by dlib.

output - directory where the classified images (facial recognition) are saved.

production - directory of images that will be analyzed for classification. It can contain multiple faces.

test - directory of images for model testing. These images SHOULD HAVE ONLY ONE FACE and the file should start with face name in the format "facename_xxxxx.jpeg". The image extent can be changed in the program.

train - directory of images for model training. These images SHOULD ALSO HAVE ONLY ONE FACE and the file should start with face name in the format "facename_xxxxx_xxxxx.jpeg". The image extent can be changed in the program.
