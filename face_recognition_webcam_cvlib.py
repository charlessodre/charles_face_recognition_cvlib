# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: This program is responsible to classify an image captured by the webcam.
# Note: Based at work of Arun Ponnusamy. Visit: https://www.arunponnusamy.com.
# Also in course "Reconhecimento de Faces e de Objetos com Python e Dlib" of Jones Granatyr (https://iaexpert.com.br/) in Udemy


# import necessary packages
import cvlib as cv
import cv2
import os
import dlib
import numpy as np
import myhelper


# ----------------------- Function Definitions -----------------------

# Print points of face in frame
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

# minimum distance reached for to attribute frame name
distance_threshold = 0.45

# Default is 05 descriptors
five_descriptors = True

save_video = True

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

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam!")
    exit()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(webcam.get(3))
frame_height = int(webcam.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_webcam = cv2.VideoWriter('output_webcam.avi',fourcc, 20.0, (frame_width,frame_height))

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for face, conf in zip(face, confidence):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)

        # create dlib face object
        face_dlib = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

        # get point of face
        points = points_detector(frame, face_dlib)

        # Print points of face in frame
        points_print(frame, points)

        # get descriptors face
        face_descriptors = face_recognition.compute_face_descriptor(frame, points)

        # convert descriptors dlib to list
        face_descriptor_list = [fd for fd in face_descriptors]

        # convert  descriptors list in numpy ndarray
        np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

        # add new column into numpy ndarray
        np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

        # calculates the Euclidean distance
        distances = np.linalg.norm(np_array_face_descriptor - file_faces_descriptors, axis=1)

        # get index minimum distance into array
        min_distance_index = np.argmin(distances)

        # get minimum distance into array
        min_distance_value = distances[min_distance_index]

        # get file name into faces_descriptors_file
        descriptor_file_name = os.path.split(faces_descriptors_file_indexes[min_distance_index])[1]

        # get frame name into descriptor_file_name
        descriptor_file_image_name = descriptor_file_name.split('_')[0]

        # classified frame name
        text_image_classified = 'unknown'

        if min_distance_value <= distance_threshold:
            text_image_classified = descriptor_file_image_name

        # put rate confidence over face. confidence that is a Face
        cv2.putText(frame, "conf {:.2f}".format(conf), (startX + 5, startY - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.7,
                    (0, 0, 255))

        # put KNN distance over frame
        cv2.putText(frame, "knn {:.2f}".format(min_distance_value), (startX + 5, startY + 25),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                    (255, 255, 0))

        # put recognized frame name  over face
        cv2.putText(frame, text_image_classified, (startX + 5, startY + 11), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    (0, 255, 0))

    # display output
    cv2.imshow("Real-time face detection", frame)

    if save_video:
        # write the flipped frame
        out_webcam.write(frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
out_webcam.release()

cv2.destroyAllWindows()

print(" ----------------- End Program -----------------")
