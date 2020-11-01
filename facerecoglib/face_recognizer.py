import glob
import os
import cv2
import numpy as np
from keras.models import load_model
from scipy import spatial
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict,Counter




class recognize():
    def __init__(self,model,graph,tolerance):
        # testx,testy = load_face("val")
        self.model = model
        self.graph = graph
        self.tolerance = tolerance
        self.dataface = defaultdict(list)
        self.threshold = 0.005
        # convert each face in the train set to an embedding

    def get_embedding(self, face_pixels):
        #face_pixels = face_pixels.astype('float32')
        #mean, std = face_pixels.mean(), face_pixels.std()
        #face_pixels = (face_pixels - mean) / std
        face_pixels = cv2.resize(face_pixels, (96,96))
        face_pixels = img_to_array(face_pixels)
        sampels = np.expand_dims(face_pixels, axis=0)
        sampels = preprocess_input(sampels)
        with self.graph.as_default():
            yhat = self.model.predict(sampels)[0,:]
        return yhat

    def normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def train(self, tx, ty):
        trainx = tx
        trainy = ty
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        idx = 0
        for face_pixels in trainx:
            embedding = self.get_embedding(face_pixels)
            self.dataface[trainy[idx]].append(embedding)
            idx += 1

    def most_frequence(self,List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)

    def findEuclideanDistance(self,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def recognize_face(self,face_input):
        face_input = self.get_embedding(face_input)
        name = []
        temp = []
        ret = 0
        for dataname,dataface in self.dataface.items():
            for idx in dataface:
                l2distance = spatial.distance.cosine(face_input,idx)
                #print('Euclidean distance from {} is {}'.format(dataname,l2distance))
                if l2distance < self.threshold:
                    name.append([dataname,l2distance])
        name = sorted(name, key=lambda l: l[1])
        if len(name) >= self.tolerance:
            for x in name:
                temp.append(x[0])
            a = temp[:self.tolerance]
            res = self.most_frequence(a)
            if res[0][1] == self.tolerance:
                ret = self.out_encoder.inverse_transform([res[0][0]])
        return ret

