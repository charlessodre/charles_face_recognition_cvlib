# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: This program is responsible for evaluating the performance of the model trained by the "face_recognition_cvlib_train.py".
# This program provides some statistics about face recognition.
# Obs: Based at work of Arun Ponnusamy. Visit: https://www.arunponnusamy.com.
# Also in course "Reconhecimento de Faces e de Objetos com Python e Dlib" of Jones Granatyr (https://iaexpert.com.br/) in Udemy


# import necessary packages
import cvlib as cv
import cv2
import os
import dlib
import numpy as np
import time
import myhelper


# ----------------------- Function Definitions -----------------------

# Show face points in image
def points_print(image_obj, points_face):
    for p in points_face.parts():
        cv2.circle(image_obj, (p.x, p.y), 2, (51, 255, 187), 1)

#  ----------------------- Global Variables -----------------------

# dlib resources path
path_dlib_resources = "resources/dlib_resources/"

# resources path
resources_path = "resources"

# test images path
path_files = "source_img/test"

# classified images path
output_path = "source_img/output/"

# file extension considered
file_extension = "*.jpeg"

# count total faces detected
total_detected_faces = 0

# count images correctly classified
total_correctly_classified_images = 0

# count images not classified
total_not_classified_images = 0

# wrong classified images (dictionary)
images_classified_wrong = {}

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

# ----------------------- Debug Variables --------------------

debug_show_all_image_window = False
debug_show_rate_confidence_image = True
debug_show_image_name = True
debug_log_classifier_error = False
debug_show_rate_KNN_distance = True
debug_print_log_KNN_distance = False
debug_show_image_window_error_classifier = False
debug_print_log_faces_count = False
debug_save_image_classified = True

# ----------------------- Begin program -----------------------

begin_test = time.time()
print(" ----------------- Begin Test: {} -----------------".format(myhelper.get_current_hour()))

list_images = myhelper.get_files(path_files, file_extension)

if len(list_images) == 0:
    print("No images found!")
    exit()

for image_file in list_images:

    # indicates whether the image was correctly classified
    correctly_classified_image = False

    # get image name into current_file
    current_file_name = image_file.split('/')[-1]

    # get image name into current_file_name
    current_file_image_name = current_file_name.split('_')[0]

    # read input image
    image = cv2.imread(image_file)

    # --------------- Begin face detection block

    # apply face detection
    faces, confidences = cv.detect_face(image)

    # loop through detected faces
    for face, conf in zip(faces, confidences):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)

        # count total detected faces in image
        total_detected_faces += 1

        # --------------- End face detection block

        # --------------- Begin face recognition block

        # create dlib face object
        face_dlib = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

        # get face points
        points = points_detector(image, face_dlib)

        # print face point in image
        points_print(image, points)

        # get descriptors face (get the main characteristics from face (with CNN))
        face_descriptors = face_recognition.compute_face_descriptor(image, points)

        # convert descriptors dlib to list
        face_descriptor_list = [fd for fd in face_descriptors]

        # convert descriptors list in numpy ndarray (this array will be storage the image and your descriptors)
        np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

        # add new column into numpy ndarray
        np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

        # calculates Euclidean distance
        distances = np.linalg.norm(np_array_face_descriptor - file_faces_descriptors, axis=1)

        # get index minimum distance from array
        min_distance_index = np.argmin(distances)

        # get minimum distance from array
        min_distance_value = distances[min_distance_index]

        # --------------- End face recognition block

        # --------------- Begin Face Recognition Comparison block

        # get file name into faces_descriptors_file
        descriptor_file_name = os.path.split(faces_descriptors_file_indexes[min_distance_index])[1]

        # get image name into descriptor_file_name
        descriptor_file_image_name = descriptor_file_name.split('_')[0]

        if debug_print_log_KNN_distance:
            print("Minimum KNN Distance: {:.2f}. Current Image: {}. Descriptor Image: {} ".format(min_distance_value,
                                                                                                  current_file_name,
                                                                                                  descriptor_file_name))

        # classified image name
        text_image_classified = 'unknown'

        # check if image was classified correctly
        if current_file_image_name == descriptor_file_image_name:
            correctly_classified_image = True
            total_correctly_classified_images += 1
            text_image_classified = descriptor_file_image_name
        elif text_image_classified == 'unknown':
            # count images not classified
            total_not_classified_images += 1
        else:
            images_classified_wrong[image_file] = [current_file_image_name, descriptor_file_image_name]

        # --------------- End Face Recognition Comparison block

        # --------------- Begin show image info block

        # put rate confidence over face. confidence that is a Face
        if debug_show_rate_confidence_image:
            cv2.putText(image, "conf {:.2f}".format(conf), (startX + 5, startY - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7,
                        (0, 0, 255))

        # put KNN distance over image
        if debug_show_rate_KNN_distance:
            cv2.putText(image, "knn {:.2f}".format(min_distance_value), (startX + 5, startY + 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 255, 0))

        # put recognized image name  over face
        if debug_show_image_name:
            cv2.putText(image, text_image_classified, (startX + 5, startY + 11), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                        (0, 255, 0))

        # --------------- End show image info block

    # --------------- Begin debug block
    # count detected faces in image
    detected_faces = len(faces)

    # --------------- Begin debug block

    if debug_print_log_faces_count and detected_faces > 1:
        print("Detected Faces in image: Faces [{}]. File: {}".format(detected_faces, image_file))

    if debug_print_log_faces_count and detected_faces == 0:
        print("No detected Faces in image. File: {}".format(image_file))

    if debug_show_image_window_error_classifier and not correctly_classified_image:
        cv2.imshow("Image was not correctly classified", image)
        cv2.waitKey()

    if debug_save_image_classified:
        # save image with recognize name
        cv2.imwrite(output_path + current_file_name, image)

    # check if image window will be show
    if debug_show_all_image_window:
        # display output
        cv2.imshow("Face Recognized", image)
        # press any key to close window
        cv2.waitKey()

    # --------------- End debug block

print("\n----------------------------- Classification Report -----------------------------\n")

print("Total processed images: {}.".format(len(list_images)))
print("Total detected faces: {}.".format(total_detected_faces))
print("Percentage detected faces: {}%.".format(total_detected_faces / len(list_images) * 100))
print("Total correctly classified faces: {}.".format(total_correctly_classified_images))
print("Total wrong classified faces: {}.".format(len(images_classified_wrong)))
print("Total not classified faces: {}.".format(total_not_classified_images))
print("Percentage correctly classified faces: {:.2f}%.".format(
    total_correctly_classified_images / total_detected_faces * 100))

print("Overall performance: {:.2f}%.".format(total_correctly_classified_images / len(list_images) * 100))

print("\n-------------------------------------------------------------------------------------")

print("\n----------------------------- Classified Error  -----------------------------\n")

for k, v in images_classified_wrong.items():
    print("image name real and classified: {} | image file: {}".format(v, k))

print("\n-------------------------------------------------------------------------------------")

end_test = time.time()
print("----------------- End Test: {} -----------------".format(myhelper.get_current_hour()))

print("----------------- Time elapsed: {} -----------------".format(myhelper.format_seconds_hhmmss(end_test - begin_test)))

# release resources
cv2.destroyAllWindows()
