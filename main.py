from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import requests
import numpy as np

app = Flask(__name__)
api = Api(app)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())


class HelloWorld (Resource):
    def get(self):
        return {"Hello": "world"}, 201


class PeopleCounter (Resource):
    def get(self):
        img = cv2.imread('bridge.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {"count": len(boxes)}


class UrlPeopleCounter (Resource):
    def get(self):
        urlParam = request.args.get('url', '')
        if not urlParam:
            return {"error": "Url param is missing or empty"}, 400
        urlResponse = requests.get(urlParam)
        if urlResponse.status_code != 200:
            return {"error": "Failed to load image"}, 400
        image_array = np.frombuffer(urlResponse.content, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {"count": len(boxes)}


class PostPeopleCOunter (Resource):
    def post(self):
        if 'image' not in request.files:
            return {"error": "No image provided"}, 400
        image_file = request.files['image'].read()
        image_array = np.frombuffer(image_file, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {"count": len(boxes)}


api.add_resource(HelloWorld, '/test')
api.add_resource(PeopleCounter, '/counter')
api.add_resource(UrlPeopleCounter, '/url-counter')
api.add_resource(PostPeopleCOunter, '/post-counter')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
