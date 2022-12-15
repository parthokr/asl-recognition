import os
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class ASL:
    def __int__(self):
        pass

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, result

    def draw_landmarks(self, image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                       mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

    def run_cv(self):
        """ tentative purpose """
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holisitc:
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                image, results = self.mediapipe_detection(frame, holisitc)
                self.draw_landmarks(image, results)
                cv2.imshow('OpeCV Feed', image)
                try:
                    # print(f'pose: {len(results.pose_landmarks.landmark)}')
                    print(f'lh: {len(results.left_hand_landmarks.landmark)}')
                    print(f'rh: {len(results.right_hand_landmarks.landmark)}')
                except:
                    pass
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        return results

    def save_data(self):
        DATA_PATH = 'MP_Data'
        actions = np.array(["hello", "thanks", "iloveyou"])
        no_sequences = 30
        sequence_length = 30
        start_folder = 0
        for action in actions:
            for sequence in range(no_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except:
                    pass

        cap = cv2.VideoCapture(0)
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # NEW LOOP
            # Loop through actions
            for action in actions:
                # Loop through sequences aka videos
                for sequence in range(start_folder, start_folder + no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(sequence_length):

                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        self.draw_landmarks(image, results)

                        # NEW Apply wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                        (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(500)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                        (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    asl = ASL()
    # asl.run_cv()
    # asl.save_data()
