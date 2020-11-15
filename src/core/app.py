from flask_restful import reqparse, abort, Api, Resource
from flask import jsonify, make_response
from flask import Flask, request
from HardHat_detection_yolov5 import Hardhat_detection_yolov5
import numpy as np
import cv2

import time
import base64

flask_app = Flask(__name__)
api = Api(app = flask_app)

class HardHat_analyzer(Resource):
    def __init__(self):
        self.hardhat_detector = Hardhat_detection_yolov5(view=False)

    def post(self):
        self.output = {}
        self.output['success'] = True

        frame_start_time = time.time()

        # Decode image_string
        nparr = np.frombuffer(request.files['image'].read(), np.uint8)
        img_ori = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # decode image

        meta_information = request.form['meta_information']

        decoding_time = time.time() - frame_start_time
        print('Decoding FPS:', 1.0 / decoding_time)

        # Inference part
        bboxes_xyx2y2, labels = self.hardhat_detector.predict(img_ori)

        # Inference part CLASSIFIER
        clasps_list = []
        glasses_list = []
        for num in range(len(bboxes_xyx2y2)):
            bbox = bboxes_xyx2y2[num]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            crop = img_ori[ymin:ymax, xmin:xmax]
            clasp, glasses = self.clasp_glasses_classifier.predict(crop)
            clasps_list.append(clasp)
            glasses_list.append(glasses)



        # Output Json construction
        heads_amount = len(bboxes_xyx2y2)
        self.output['heads_amount'] = heads_amount
        self.output['heads'] = []
        self.output['meta_information'] = meta_information
        for num in range(len(bboxes_xyx2y2)):
            xyx2y2 = bboxes_xyx2y2[num]
            head_json = {}
            head_json['bbox'] = {}
            head_json['bbox'] = {'xmin': int(xyx2y2[0]), 'ymin': int(xyx2y2[1]), 'xmax': int(xyx2y2[2]), 'ymax': int(xyx2y2[3])}
            head_json['label'] = labels[num]
            self.output['heads'].append(head_json)

        total_time = time.time() - frame_start_time
        print('Total FPS:', 1.0 / total_time)

        return make_response(self.output, 200)

api.add_resource(HardHat_analyzer, '/', methods=['GET', 'POST'])
if __name__ == '__main__':
    flask_app.run(debug=False, host='0.0.0.0', port=7778, threaded=True)

