import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from collections import OrderedDict
import numpy as np
from pyimagesearch.centroidtracker import CentroidTracker
import datetime
from utils import label_map_util
from utils import visualization_utils as vis_util
import tensorflow as tf
import imgconvert

class result_person():
    def __init__(self,status,id):
        self.stat= status
        self.id = id


class classifier():
    def __init__(self,apdmod):
        self.ctnotsafe = CentroidTracker(maxDisappeared=80, maxDistance=60)
        self.objectnotsafe = {}
        self.tempworker = []
        self.model = apdmod
        PATH_TO_CKPT = "inference_graph/frozen_inference_graph.pb"
        PATH_TO_LABELS = "training/mscoco_label_map.pbtxt"
        NUM_CLASSES = 1
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        time = datetime.datetime.now()
        self.lsttime = time.strftime("%d")
        self.lastnextid = 0


    def predictin(self,imgin):
        s = cv2.resize(imgin, (224, 224))
        img = np.array(s)
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        output = self.model.predict(img)
        if max(output[0]) > 0.95:
            if output[0][0] > output[0][1]:
                return "safe"
            elif output[0][1] > output[0][0]:
                return 'notsafe'
        else:
            return "waiting"


    def worker_object(self,id,image_worker,date,time):
        work_object = {
            "id_worker": id,
            "image_worker": image_worker,
            "date": date,
            "time_stamp": time
        }
        return work_object



    def procbody(self,image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        temp=[]
        res = []
        rectnotsafe=[]
        #reset counter
        time = datetime.datetime.now()
        cttime = time.strftime("%d")
        if cttime != self.lsttime:
            self.ctnotsafe.nextObjectID = 0
            self.lsttime = cttime
        im_input= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(im_input, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        coordinate = vis_util.return_coordinates(
            im_input,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=5,
            min_score_thresh=0.5,
            skip_scores=True)

        for box in coordinate:
            xmin = box[0]
            xmax = box[1]
            ymin = box[2]
            ymax = box[3]
            st = box[4]
            status = st[0]

            if status == "person":
                temp.append([xmin,ymin,xmax,ymax])
                feed = image[ymin:ymax, xmin:xmax]
                idlabel = 0
                apdstat = self.predictin(feed)
                if apdstat == "safe":
                    color = (0,255,0)
                elif apdstat == "waiting":
                    color = (255,255,255)
                else:
                    rectnotsafe.append((xmin, ymin, xmax, ymax))
                    color = (0,0,255)
                res.append(result_person(apdstat, idlabel))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, "APD Status = {}".format(apdstat), (xmin, ymin), font, 1,
                            color, 2)
        #update counter worker_foul
        self.objectnotsafe = self.ctnotsafe.update(rectnotsafe)
        self.notsafenextid = self.ctnotsafe.nextObjectID
        if self.notsafenextid != self.lastnextid:
            date = datetime.datetime.now().strftime('%Y-%m-%d')
            ts = datetime.datetime.now().strftime('%H:%M')
            binimg = imgconvert.ImgToBin(image)
            data = self.worker_object(self.notsafenextid,binimg,date,ts)
            self.tempworker.append(data)
            self.lastnextid = self.notsafenextid
        return res,temp

