# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: This program is for the training to extract the main characteristics of the images. This program generates the files with the descriptors of the images.
# Note: Based at work of Arun Ponnusamy. Visit: https://www.arunponnusamy.com.
# Also in course "Reconhecimento de Faces e de Objetos com Python e Dlib" of Jones Granatyr (https://iaexpert.com.br/) in Udemy


# import necessary packages
import cvlib as cv
import cv2
import os
import dlib
import numpy as np
import _pickle as pickle
import time
import myhelper

# ----------------------- Function Definitions -----------------------

# show face points in image
def points_print(image_obj, points_face):
    for p in points_face.parts():
        cv2.circle(image_obj, (p.x, p.y), 2, (51, 255, 187), 1)

#  ----------------------- Global Variables -----------------------


# dlib resources
path_dlib_resources = "resources/dlib_resources/"

# train file path
path_files = "source_img/train"

# File extension considered
file_extension = "*.jpeg"

# resource file
resources_path = "resources"

# dictionary of index images file trained.
# Used to identify the name of the image and its descriptors
index_dict = {}

index = 0

# faces descriptor values
faces_descriptor = None

# Total detected faces in image
total_detected_faces = 0

# Default is 05 descriptors
five_descriptors = True

if five_descriptors:
    resource_file_descriptors = "image_descriptors_05.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_05.pickle"
else:
    resource_file_descriptors = "image_descriptors_68.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_68.pickle"

# ----------------------- Resources Dlib -------------------------------------------

if five_descriptors:
    # face point detector with 05 points
    predictor_shape_file = "shape_predictor_5_face_landmarks.dat"
else:
    # face point detector with 68 points
    predictor_shape_file = "shape_predictor_68_face_landmarks.dat"

points_detector = dlib.shape_predictor(os.path.join(path_dlib_resources, predictor_shape_file))

# trained file of Convolutional Neural Network (CNN)
face_recognition = dlib.face_recognition_model_v1(
    os.path.join(path_dlib_resources, "dlib_face_recognition_resnet_model_v1.dat"))


# ----------------------- Begin program -----------------------

begin_train = time.time()

print(" ----------------- Begin Train: {} -----------------".format(myhelper.get_current_hour()))

list_images = myhelper.get_files(path_files, file_extension)

if len(list_images) == 0:
    exit()

for image_file in list_images:

    # read input image
    image = cv2.imread(image_file)

    # --------------- Begin face detection block

    # apply face detection
    faces, confidences = cv.detect_face(image)

    # count total detected faces in image
    total_detected_faces = len(faces)

    # check if only one face was detected in image
    if total_detected_faces == 1:

        # loop through detected faces
        for face, conf in zip(faces, confidences):
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

    # --------------- End face detection block

            # --------------- Begin face recognition block

            # create dlib face object
            face_dlib = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

            # get face points
            points = points_detector(image, face_dlib)

            # print face points in image
            points_print(image, points)

            # get descriptors face (get the main characteristics from face (with CNN))
            face_descriptors = face_recognition.compute_face_descriptor(image, points)

            # print("\n--------- Image file: {} ---------". format(image_file))
            # print("Image Points Detected: {}".format(len(face_descriptors)))

            # convert descriptors dlib to list
            face_descriptor_list = [fd for fd in face_descriptors]

            # convert descriptors list in numpy ndarray (this array will be storage the image and your descriptors)
            np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

            # add new column into numpy ndarray
            np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

            if faces_descriptor is None:
                faces_descriptor = np_array_face_descriptor

            else:
                faces_descriptor = np.concatenate((faces_descriptor, np_array_face_descriptor), axis=0)

            # used to identify the name of the image and its descriptors
            index_dict[index] = image_file

            index += 1

            # --------------- End face recognition block

    elif total_detected_faces > 1:
        print("More than one face was detected in image: Faces [{}]. File: {}".format(total_detected_faces,
                                                                                      image_file))
    else:
        print("No face was detected in image. File: {}".format(image_file))

    # --------------- End face detection block

# save image descriptor training file
np.save(os.path.join(resources_path, resource_file_descriptors), faces_descriptor)

# save image descriptor indexes training file
with open(os.path.join(resources_path, resource_file_descriptors_indexes), 'wb') as f:
    pickle.dump(index_dict, f)

print(" ----------------- Info Train -----------------")
print("Images Analysed: {}".format(len(list_images)))
print("Image descriptor training file length: {}. Shape: {}".format(len(faces_descriptor), faces_descriptor.shape))

end_train = time.time()
print("\n----------------- End Train: {} -----------------".format(myhelper.get_current_hour()))

print("----------------- Time elapsed: {} -----------------".format(myhelper.format_seconds_hhmmss(end_train - begin_train)))

# release resources
cv2.destroyAllWindows()
