import os
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from string import ascii_uppercase

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class ASL:
    def __init__(self):
        self.model = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.y = None
        self.x = None
        self.actions = np.array([x for x in ascii_uppercase])
        self.DATA_PATH = 'MP_Data_Kaggle'
        self.no_sequences = 30
        self.sequence_length = 30
        self.start_folder = 0

        self.no_of_dataset_per_letter_in_kaggle = 70

        self.log_dir = os.path.join("logs")
        self.tb_callback = TensorBoard(log_dir=self.log_dir)

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))


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
                    print(results.pose_landmarks.landmark)
                    # print(f'lh: {len(results.left_hand_landmarks.landmark)}')
                    # print(f'rh: {len(results.right_hand_landmarks.landmark)}')
                except:
                    pass
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        return results

    def extract_keypoints(self, results: NamedTuple) -> NamedTuple:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4)

        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
            if results.face_landmarks else np.zeros(468 * 3)

        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)

        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose, face, left_hand, right_hand])

    def save_data(self):
        # for action in self.actions:
        #     for sequence in range(self.no_sequences):
        #         try:
        #             os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
        #         except:
        #             pass
        for action in self.actions:
            dirmax = np.max(np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int))
            for sequence in range(1, self.no_sequences + 1):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(dirmax + sequence)))
                except:
                    pass

        cap = cv2.VideoCapture(0)
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # NEW LOOP
            # Loop through actions
            for action in self.actions:
                # Loop through sequences aka videos
                for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):

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
                        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()

    def preprocess_data(self):
        label_map = {label: num for num, label in enumerate(self.actions)}
        # print(label_map)
        features, labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                features.append(window)
                labels.append(label_map[action])

        # print(labels)
        # print(len(labels))
        # print(np.array(features).shape)

        self.x = np.array(features)
        self.y = to_categorical(labels).astype(int)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.05)

        # print(x)
        # print(y)

    def train_lstm(self):
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=2000, callbacks=[self.tb_callback])

        self.model.save('action.h5')

        print(self.model.summary())

    def eval_using_confusion_matrix(self):
        pass

    def use_kaggle_dataset(self):
        for action in self.actions:
            os.makedirs(os.path.join(self.DATA_PATH, action))

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # NEW LOOP
            # Loop through actions
            for action in self.actions:
                # Loop through video length aka sequence length
                for frame_num in range(self.no_of_dataset_per_letter_in_kaggle):
                    # Read feed
                    # ret, frame = cap.read()
                    # Read from kagggle's dataset
                    frame = cv2.imread(f"asl_dataset/{action.lower()}/hand1_{action.lower()}_bot_seg_1_cropped.jpeg")
                    # Make detections
                    image, results = self.mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    self.draw_landmarks(image, results)

                    # NEW Export keypoints
                    keypoints = self.extract_keypoints(results)
                    npy_path = os.path.join(self.DATA_PATH, action, str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

    def test(self):
        self.model.load_weights('action.h5')
        sequence = []
        sentence = []
        threshold = 0.4


        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                self.draw_landmarks(image, results)

                # Prediction logic
                keypoints = self.extract_keypoints(results)
                sequence.insert(0, keypoints)
                sequence = sequence[:30]

                if len(sequence) == 30:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.actions[np.argmax(res)])

                # Render to screen
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    asl = ASL()
    # asl.run_cv()
    # asl.save_data()
    asl.use_kaggle_dataset()
    # asl.preprocess_data()
    # asl.train_lstm()
    # asl.test()