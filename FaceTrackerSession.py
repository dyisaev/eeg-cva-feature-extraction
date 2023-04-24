import face_recognition
import cv2
from datetime import datetime
import time
import os
import numpy as np
import json


class FaceTrackerSession:

    def __init__(self, username, log_path):
        self.user = username

#        if not os.path.isfile(log_path):
#            raise Exception("Log file path invalid: {}".format(log_path))
#        else:
        self.log_path = log_path

        # Set some parameters
        self.buffer_size = 5    # tracking buffer size
        self.scale = .4         # image rescaling
        self.tolerance = 0.5    # face recognition confidence threshold
        
        # Face encoding and label arrays
        self.face_encodings = []
        self.face_labels = []

    def write_log(self, text):
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                log_text = ",".join([str(datetime.now()), self.user, text]) + "\n"
                f.write(log_text)

    def check_face_overlap(self, face_locations, face_position_thresh, face_size_thresh):    
        overlapping_face_centers = [self.get_face_center(face_locations[0])]
        overlapping_face_sizes = [self.get_face_size(face_locations[0])]
        for face_location in face_locations[1:]:
            face_center = self.get_face_center(face_location)
            face_size = self.get_face_size(face_location)
            if self.check_face_position(face_center, overlapping_face_centers, face_position_thresh) and \
            self.check_face_size(face_size, overlapping_face_sizes, face_size_thresh):
                overlapping_face_centers.append(face_center)
                overlapping_face_sizes.append(face_size)
            else:
                return False
        
        return True

    @staticmethod
    def check_face_position(center, centers, thresh):
        mean = np.mean(centers, axis=0)
        d = np.sqrt(np.sum(np.square(center - mean)))
        if d < thresh:
            return True
        else:
            return False

    @staticmethod
    def check_face_size(size, sizes, thresh):
        mean = np.mean(sizes)
        d = np.abs(size - mean)
        if d < thresh:
            return True
        else:
            return False

    @staticmethod
    def get_face_center(face_location):
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]

        x = 0.5 * (left + right)
        y = 0.5 * (top + bottom)

        return np.array([[x, y]])

    @staticmethod
    def get_face_size(face_location):
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]

        w = right - left
        h = top - bottom

        return max(w, h)

    @staticmethod
    def verify_face(frame, face_location):
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]
        copy = frame.copy()
        
        cv2.rectangle(copy, (left, top), (right, bottom), (0, 0, 255), 2)

        print("Is this the target's face? y/n, press 'q' to quit")
        cv2.imshow('Face Verification', copy)
        while True:
            key = cv2.waitKey(0) 
            if key & 0xFF == ord('y'):
                cv2.destroyAllWindows()
                return True
            if key & 0xFF == ord('n'):
                cv2.destroyAllWindows()
                return False
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                raise Exception("Early termination requested")

    def process_video(self, input_video, output_video, output_json):
        # Open the input movie file
        videocapture = cv2.VideoCapture(input_video)
        if not videocapture.isOpened():
            raise Exception("Cound not open {}".format(input_video))
        
        # Read video specs
        video_length = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = videocapture.get(cv2.CAP_PROP_FPS)

        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))

        # Initialize buffers for face tracking
        target_face_centers = np.full((self.buffer_size, 2), np.nan)
        target_face_sizes = np.full(self.buffer_size, np.nan)

        # Output face location to json
        face_output = {"frame": [],
                    "left": [],
                    "right": [],
                    "top": [],
                    "bottom": [],
                    "total_frames": video_length}

        # Set some parameters
        face_size_thresh = min(video_height, video_width) * self.scale / 8
        face_position_thresh = min(video_height, video_width) * self.scale / 8

        frame_number = 0

        while True:
            # Grab a single frame of video
            ret, frame = videocapture.read()

            # Quit when the input video file ends
            if not ret:
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            t1 = time.time()
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            t2 = time.time()
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            t3 = time.time()
            print("Face detection time: {}s".format(t2-t1))
            print("Face encoding time: {}s".format(t3-t2))

            is_target = []
            target_face_center = np.full((1, 2), np.nan)
            target_face_size = np.nan
            num_target_face = 0
            num_other_face = 0
            buffer_index = (frame_number-1) % self.buffer_size
            min_face_distance = 1
            
            t1 = time.time()
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compute face center and size
                face_center = self.get_face_center(face_location)
                face_size = self.get_face_size(face_location)
                # print("Face center: {}".format(face_center))
                # print("Face size: {}".format(face_size))

                # Compute the distance of the face to known faces
                face_distance = face_recognition.face_distance(self.face_encodings, face_encoding)
                if face_distance.size > 0:
                    min_face_distance = np.amin(face_distance)
                    face_label = self.face_labels[np.argmin(face_distance)]

                # Check confidence of match
                if min_face_distance < self.tolerance:
                    if face_label == "target":
                        is_target.append(True)
                        # Save the center of target's face
                        target_face_center = face_center
                        target_face_size = face_size
                        num_target_face += 1
                    elif face_label == "other":
                        is_target.append(False)
                        num_other_face += 1
                    else:
                        raise Exception("Unknown face label {}".format(face_label))
                else:
                    # Check if target's face has been tracked
                    if any(np.isnan(target_face_sizes)):
                        # Manually specify if face is target / other
                        is_target.append(self.verify_face(rgb_small_frame[:, :, ::-1], face_location))
                    else:
                        # Check face position and size
                        if not self.check_face_position(face_center, target_face_centers, face_position_thresh):
                            # Face position does not match, not the target's face
                            is_target.append(False)
                        else:
                            if self.check_face_size(face_size, target_face_sizes, face_size_thresh):
                                # Both face position and size match, is the target's face
                                is_target.append(True)
                                print("Face position and size matches for target")
                            else:
                                # Manually specify if face is target / other
                                is_target.append(self.verify_face(rgb_small_frame[:, :, ::-1], face_location))

                    # Add encoding and labels to known list
                    self.face_encodings.append(face_encoding)
                    if is_target[-1]:
                        self.face_labels.append("target")
                        target_face_center = face_center
                        target_face_size = face_size
                        num_target_face += 1
                    else:
                        self.face_labels.append("other")
                        num_other_face += 1

            # Handle 2 cases where more than 1 target's face is detected
            if num_target_face > 1: 
                # Case 1: Overlapping target face detections
                target_face_locations = [f for f, i in zip(face_locations, is_target) if i]
                is_overlapping = self.check_face_overlap(target_face_locations, face_position_thresh, face_size_thresh)
                if is_overlapping:
                    # Use first detected target face
                    did_set_target = False
                    for index in range(len(face_locations)):
                        if is_target[index]:
                            if not did_set_target:
                                face_location = face_locations[index]
                                target_face_center = self.get_face_center(face_location)
                                target_face_size = self.get_face_size(face_location)
                                did_set_target = True
                            else:
                                is_target[index] = False
                                num_target_face -= 1
                else:
                    # Case 2: Incorrect target face detections
                    print("More than 1 target's face detected, please verify")
                    for index in range(len(face_locations)):
                        if is_target[index]:
                            face_location = face_locations[index]
                            face_encoding = face_encodings[index]
                            # Manually specify if face is target / other
                            is_target[index] = self.verify_face(rgb_small_frame[:, :, ::-1], face_location)
                            if is_target[index]:
                                target_face_center = self.get_face_center(face_location)
                                target_face_size = self.get_face_size(face_location)
                            else:
                                # Add incorrect encoding to known list
                                self.face_encodings.append(face_encoding)
                                self.face_labels.append("other")
                                num_target_face -= 1

            # Update tracked face position and size
            target_face_centers[buffer_index, :] = target_face_center
            target_face_sizes[buffer_index] = target_face_size

            t2 = time.time()
            print("Face recognition time: {}s".format(t2-t1))
            print("{} target face(s) found, {} other face(s) found".format(num_target_face, num_other_face))

            # Label the video frame with face bounding boxes
            # t1 = time.time()
            for (top, right, bottom, left), target_flag in zip(face_locations, is_target):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top = int(top / self.scale)
                right = int(right / self.scale)
                bottom = int(bottom / self.scale)
                left = int(left / self.scale)

                # name = None
                color = None
                if target_flag:
                    # name = "Target"
                    color = (0, 255, 0)
                    face_output["frame"].append(frame_number)
                    face_output["left"].append(left)
                    face_output["top"].append(top)
                    face_output["right"].append(right)
                    face_output["bottom"].append(bottom)
                else:
                    # name = "Other"
                    color = (0, 0, 255)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)    

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX            
                # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # t2 = time.time()
            # print("Drawing time: {}s".format(t2-t1))

            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number+1, video_length))
            # t1 = time.time()
            videowriter.write(frame)
            # t2 = time.time()
            # print("Write frame time: {}s".format(t2-t1))

            # Commment the following if you don't want the program to crash
    #        if num_target_face > 1:               
    #            raise Exception("More than one target's face found for frame {}".format(frame_number))

            frame_number += 1

        # All done!
        videocapture.release()
        cv2.destroyAllWindows()

        # Write json
        with open(output_json, 'w') as outfile:
            json.dump(face_output, outfile)

        # Write log
        self.write_log(input_video)
