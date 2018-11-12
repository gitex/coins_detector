#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np
import cv2
import glob
import os

MAIN_PATH = os.path.abspath(os.path.dirname(__file__))


class Coin:
    def __init__(self, name, title, value, folder_name):
        self.name = name
        self.title = title
        self.value = value
        self.folder_name = folder_name

    def __str__(self):
        return "<class '{}'>".format(self.name)


class CoinDetector:
    def __init__(self, input_image):
        self.input_image = os.path.abspath(input_image)

        self.image = cv2.imread(self.input_image)
        self.output = self.image.copy()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gray = clahe.apply(gray)

        # Классификатор
        self.clf = MLPClassifier(solver="lbfgs")
        self.score = None
        self.coins = None

        # Подсчет
        self.count_coins = 0
        self.total_value = 0


    def calculate_histogram(self, img):
        m = np.zeros(img.shape[:2], dtype="uint8")
        (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.circle(m, (w, h), 60, 255, -1)

        h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # return normalized "flattened" histogram
        return cv2.normalize(h, h).flatten()

    def calculate_histogram_from_file(self, filename):
        img = cv2.imread(filename)
        return self.calculate_histogram(img)

    def upload_coins(self, coins, folder=MAIN_PATH):
        x = []
        y = []

        self.coins = coins

        for coin in coins:
            for example_file in glob.glob(os.path.join(folder, coin.folder_name, "*")):
                x.append(self.calculate_histogram_from_file(example_file))
                y.append(coin.name)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2)

        self.clf.fit(X_train, y_train)
        self.score = int(self.clf.score(X_test, y_test) * 100)

    def predict_coin(self, part_of_image):
        hist = self.calculate_histogram(part_of_image)
        s = self.clf.predict([hist])
        return self.get_coin_by_name(s)

    def get_coin_by_name(self, name):
        for coin in self.coins:
            if name == coin.name:
                return coin

        return None

    def draw_predictions(self):
        diameter = []
        predicted_coins = []
        coordinates = []

        blurred = cv2.GaussianBlur(self.gray, (7, 7), 0)

        # Находим круги
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
            param1=200, param2=100, minRadius=50, maxRadius=120)

        if circles is not None:
            for (x, y, r) in circles[0, :]:
                diameter.append(r)

            circles = np.round(circles[0, :]).astype("int")

            for (x, y, d) in circles:
                self.count_coins += 1

                coordinates.append((x, y))

                part = self.image[y - d:y + d, x - d:x + d]

                predicted_coin = self.predict_coin(part)
                predicted_coins.append(predicted_coin)


                cv2.circle(self.output, (x, y), d, (0, 255, 0), 2)
                cv2.putText(self.output, predicted_coin.name,
                            (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Считаем и записываем значения
        for i in range(self.count_coins):
            d = diameter[i]
            coin = predicted_coins[i]
            (x, y) = coordinates[i]

            self.total_value += coin.value

            cv2.putText(
                self.output, coin.title,
                (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    def save(self, output_file):
        cv2.putText(self.output, "Coins detector",
                    (5, self.output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(self.output, "Coins detected: {}, total: {:2}".format(self.count_coins, self.total_value),
                    (5, self.output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(self.output, "Classifier mean accuracy: {}%".format(self.score),
                    (5, self.output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)

        cv2.imwrite(output_file, self.output)


if __name__ == "__main__":
    detector = CoinDetector('D:\YandexDisk\PycharmProjects\OpenCV_Projects\Coins\input1.jpg')

    coins = []

    for title, v in [('One', 1), ('Two', 2), ('Five', 5), ('Ten', 10)]:
        coins.append(Coin(title, str(v), v, str(v)))

    detector.upload_coins(coins=coins, folder=os.path.join(MAIN_PATH, "detector", "examples"))

    detector.draw_predictions()
    detector.save('output_coins.jpg')
