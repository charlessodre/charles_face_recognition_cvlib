# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: This program is responsible to identifier a persons into images.
# After the training (face_recognition_cvlib_train.py) and testing (face_recognition_cvlib_test.py ) of the model,
# this program is responsible for batch processing to classified and save the image with the classification assigned to it.
# Obs: Based at work of Arun Ponnusamy. Visit: https://www.arunponnusamy.com.
# Also in course "Reconhecimento de Faces e de Objetos com Python e Dlib" of Jones Granatyr (https://iaexpert.com.br/) in Udemy


# import necessary packages
import cvlib as cv
import cv2
import os
import dlib
import numpy as np
import myhelper


# ----------------------- Function Definitions -----------------------

# show face points in image
def points_print(image_obj, points_face):
    for p in points_face.parts():
        cv2.circle(image_obj, (p.x, p.y), 2, (51, 255, 187), 1)


#  ----------------------- Global Variables -----------------------

# dlib resources path
path_dlib_resources = "resources/dlib_resources/"

# resources path
resources_path = "resources"

# file path
path_files = "source_img/production"

# File extension considered
file_extension = "*.jpeg"

# classified image path
output_path = "source_img/output/"

# minimum distance reached for to attribute image name
distance_threshold = 0.49

# Default is 05 descriptors
five_descriptors = True

# ----------------------- Resources Dlib -------------------------------------------

if five_descriptors:
    # face point detector with 05 points
    predictor_shape_file = "shape_predictor_5_face_landmarks.dat"
else:
    # face point detector with 68 points
    predictor_shape_file = "shape_predictor_68_face_landmarks.dat"

points_detector = dlib.shape_predictor(os.path.join(path_dlib_resources, predictor_shape_file))

# dlib trained file of Convolutional Neural Network (CNN)
face_recognition = dlib.face_recognition_model_v1(
    os.path.join(path_dlib_resources, "dlib_face_recognition_resnet_model_v1.dat"))

# ----------------------- Resources  -------------------------------------------

if five_descriptors:
    resource_file_descriptors = "image_descriptors_05.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_05.pickle"
else:
    resource_file_descriptors = "image_descriptors_68.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_68.pickle"

# image descriptors file trained for our own images
file_faces_descriptors = np.load(os.path.join(resources_path, resource_file_descriptors))

# image descriptors indexes file trained for our own images
faces_descriptors_file_indexes = np.load(os.path.join(resources_path, resource_file_descriptors_indexes))

# ----------------------- Begin program -----------------------

print(" ----------------- Begin Program -----------------")

list_images = myhelper.get_files(path_files, file_extension)

if len(list_images) == 0:
    exit(0)

for image_file in list_images:

    # get image name into current_file
    current_file_name = image_file.split('/')[-1]

    # get image name into current_file_name
    current_file_image_name = current_file_name.split('_')[0]

    # read input image
    image = cv2.imread(image_file)

    # apply face detection
    faces, confidences = cv.detect_face(image)

    # loop through detected faces
    for face, conf in zip(faces, confidences):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)

        # create dlib face object
        face_dlib = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

        # get face points
        points = points_detector(image, face_dlib)

        # print face points in image
        points_print(image, points)

        # get descriptors face (get the main characteristics from face (with CNN))
        face_descriptors = face_recognition.compute_face_descriptor(image, points)

        # convert descriptors dlib to list
        face_descriptor_list = [fd for fd in face_descriptors]

        # convert descriptors list in numpy ndarray (this array will be storage the image and your descriptors)
        np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

        # add new column into numpy ndarray
        np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

        # calculates the Euclidean distance
        distances = np.linalg.norm(np_array_face_descriptor - file_faces_descriptors, axis=1)

        # get index minimum distance from array
        min_distance_index = np.argmin(distances)

        # get minimum distance from array
        min_distance_value = distances[min_distance_index]

        # get file name into faces_descriptors_file
        descriptor_file_name = os.path.split(faces_descriptors_file_indexes[min_distance_index])[1]

        # get image name into descriptor_file_name
        descriptor_file_image_name = descriptor_file_name.split('_')[0]

        # classified image name
        text_image_classified = 'unknown'

        if min_distance_value <= distance_threshold:
            text_image_classified = descriptor_file_image_name

        # put KNN distance over image
        cv2.putText(image, "knn {:.2f}".format(min_distance_value), (startX + 5, startY + 25),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                    (255, 255, 0))

        # put recognized image name  over face
        cv2.putText(image, text_image_classified, (startX + 5, startY + 11), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    (0, 255, 0))

    # display output
    # cv2.imshow("Face Recognized", image)
    # press any key to close window
    # cv2.waitKey()

    # save image with recognize name
    cv2.imwrite(output_path + current_file_name, image)

# release resources
cv2.destroyAllWindows()

print(" ----------------- End Program -----------------")
