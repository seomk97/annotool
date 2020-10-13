import sys
import time
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# from PyQt5 import QtCore, QtGui
from main import *
import threading
# from queue import Queue

form_class = uic.loadUiType("./pjtlibs/qtui.ui")[0]

video_path = []
text = None
copied_text = None
framecount = 0
handler = True
end = False
flush = False
pause = False
objimg = np.array([])
set_speed = 1
target_only_view = False
qimg_1 = 0
qimg_2 = 0
tracking = False

YoloV3 = yolo
score_threshold = 0.5
iou_threshold = 0.45
CLASSES = YOLO_COCO_CLASSES
max_cosine_distance = 0.7
nn_budget = None

# initialize deep sort object
model_filename = "./pjtlibs/mars-small128.pb"  # deep sort 웨이트
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

NUM_CLASS = read_class_names(CLASSES)  # name strip 하는 커스텀함수 from utils
key_list = list(NUM_CLASS.keys())
val_list = list(NUM_CLASS.values())
#
# queueSize = 32
# Q = Queue(maxsize=queueSize)


class MyWindow(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.file_load)
        self.pushButton_2.clicked.connect(self.start)
        self.pushButton_3.clicked.connect(self.select)
        self.pushButton_4.clicked.connect(self.thread)
        self.pushButton_5.clicked.connect(self.w_key)
        self.pushButton_6.clicked.connect(self.r_key)
        self.pushButton_7.clicked.connect(self.q_key)
        self.pushButton_8.clicked.connect(self.s_key)
        self.pushButton_9.clicked.connect(self.space_key)
        self.pushButton_10.clicked.connect(self.target_change)
        self.pushButton_11.clicked.connect(self.speed_up)
        self.pushButton_12.clicked.connect(self.speed_down)
        self.pushButton_13.clicked.connect(self.open_folder)
        self.pushButton_14.clicked.connect(self.target_only_view)
        self.label7.setText("배속  x%d 배" % set_speed)
        self.actionQuit.triggered.connect(qApp.quit)
        self.actionQuit.setShortcut('Ctrl+Q')
        self.pushButton.setShortcut('l')
        self.pushButton_2.setShortcut('a')
        self.pushButton_3.setShortcut('o')
        self.pushButton_4.setShortcut('t')
        self.pushButton_5.setShortcut('w')
        self.pushButton_6.setShortcut('r')
        self.pushButton_7.setShortcut('q')
        self.pushButton_8.setShortcut('s')
        self.pushButton_9.setShortcut(Qt.Key.Key_Space)
        self.pushButton_10.setShortcut('c')
        self.pushButton_11.setShortcut(Qt.Key.Key_Right)
        self.pushButton_12.setShortcut(Qt.Key.Key_Left)

    def file_load(self):
        global video_path
        video_path_buffer = QFileDialog.getOpenFileName(self, None, None, "Video files (*.mp4)")
        if video_path_buffer[0] is not '' and video_path_buffer != video_path:
            video_path = video_path_buffer
            self.label.setText(video_path[0])
            self.img_load()
        else:
            pass
        if not video_path:
            self.pushButton_2.setEnabled(False)
        else:
            if video_path_buffer[0] == '':
                return
            else:
                self.pushButton_2.setEnabled(True)
                self.pushButton_7.setEnabled(True)

    def img_load(self):
        pixmap = QPixmap("./captured/frame.jpg")
        self.label2.setPixmap(pixmap)

    def start(self):
        if os.path.isdir("./captured/"):
            if os.listdir("./captured/"):
                QMessageBox.about(self, "디렉토리 존재", "파일이 이미 존재합니다")
                return
            else:
                pass
        else:
            pass

        global flush
        global pause
        flush = False
        pause = False
        self.pushButton_9.setText('Pause\n(space)')
        self.pushButton_9.setShortcut(Qt.Key.Key_Space)
        Object_tracking(yolo, video_path[0], '', input_size=input_size, show=True, iou_threshold=0.3,
                        rectangle_colors=(255, 0, 0), Track_only=["person"])
        self.img_load()
        if os.path.isfile("./captured/frame.jpg"):
            os.remove("./captured/frame.jpg")
        self.pushButton_3.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_7.setEnabled(True)

    def select(self):
        global text
        global copied_text
        global pause

        if not tracking:
            text, ok = QInputDialog.getInt(self, 'Object Select', '오브젝트 번호를 입력해주세요')
            copied_text = text
            if ok:
                self.label3.setText('obj ' + str(copied_text))
                self.label5.setText('person ' + str(text))
                self.pushButton_4.setEnabled(True)
            else:
                text = None
                self.label3.setText("None")
                self.label5.setText("None")
                self.pushButton_4.setEnabled(False)
                return

        if tracking:
            if pause:
                pass
            else:
                self.space_key()

            text_2, ok = QInputDialog.getInt(self, 'Object Select', '오브젝트 번호를 입력해주세요')
            if ok:
                text = text_2
                copied_text = text_2
                self.label3.setText('obj ' + str(text))
                self.label5.setText('person ' + str(text))
            else:
                return

    def target_change(self):
        global text
        global pause

        if pause:
            pass
        else:
            self.space_key()

        text_2, ok = QInputDialog.getInt(self, 'Object Select', '오브젝트 번호를 입력해주세요')
        if ok:
            text = text_2
            self.label5.setText('person ' + str(text))
        else:
            return

    def thread(self):
        self.pushButton_4.setEnabled(False)
        self.pushButton_10.setEnabled(True)
        self.pushButton_11.setEnabled(True)
        self.pushButton_14.setEnabled(True)
        # th2 = threading.Thread(target=self.vidload)
        # th2.setDaemon(True)
        th = threading.Thread(target=self.track)
        th.setDaemon(True)
        # th2.start()
        th.start()
        return

    def w_key(self):
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("obj 추적실패 저장안됨")
            return
        cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
        with open('./captured/obj%d/%d.txt' % (label_n_count[0], label_n_count[0]), 'a') as f:
            f.write("%d.jpg walking\n" % framecount)
        self.label4.setText("%d.jpg   walking" % framecount)
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def r_key(self):
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("obj 추적실패 저장안됨")
            return
        cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
        with open('./captured/obj%d/%d.txt' % (label_n_count[0], label_n_count[0]), 'a') as f:
            f.write("%d.jpg running\n" % framecount)
        self.label4.setText("%d.jpg   running" % framecount)
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def s_key(self):
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("추적실패")
            return
        cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
        with open('./captured/obj%d/%d.txt' % (label_n_count[0], label_n_count[0]), 'a') as f:
            f.write("%d.jpg stop\n" % framecount)
        self.label4.setText("%d.jpg   stop" % framecount)
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def space_key(self):
        global pause
        if not pause:
            pause = True
            self.pushButton_9.setText('Play\n(space)')
            self.pushButton_9.setShortcut(Qt.Key.Key_Space)
        else:
            pause = False
            self.pushButton_9.setText('Pause\n(space)')
            self.pushButton_9.setShortcut(Qt.Key.Key_Space)
        return

    def q_key(self):
        self.flush()
        return

    def flush(self):
        global text
        global end
        global flush
        global pause
        global set_speed
        global tracking
        tracking = False

        if pause:
            pass
        else:
            self.space_key()

        reply = QMessageBox.question(self, 'Message', '초기화합니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            flush = True
            set_speed = 1
            self.label.setText("Video Path")
            self.label3.setText("None")
            self.label5.setText("None")
            self.label7.setText("배속  x%d 배" % set_speed)
            self.label4.setText("")
            if os.path.isfile("./captured/frame.jpg"):
                os.remove("./captured/frame.jpg")
            # if os.path.isdir("./captured/obj%d/" % text) and text is not None:
            #     os.remove("./captured/obj%d/" % text)
            self.img_load()
            pixmap = QPixmap("./captured/frame.jpg")
            self.label6.setPixmap(pixmap)
            self.pushButton_2.setEnabled(False)
            self.pushButton_3.setEnabled(False)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
            self.pushButton_7.setEnabled(False)
            self.pushButton_8.setEnabled(False)
            self.pushButton_9.setEnabled(False)
            self.pushButton_10.setEnabled(False)
            self.pushButton_11.setEnabled(False)
            self.pushButton_12.setEnabled(False)
            self.pushButton_14.setEnabled(False)
            text = None
            end = False
        else:
            if end:
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(False)
                self.pushButton_4.setEnabled(False)
                self.pushButton_5.setEnabled(False)
                self.pushButton_6.setEnabled(False)
                self.pushButton_7.setEnabled(True)
                self.pushButton_8.setEnabled(False)
                self.pushButton_9.setEnabled(False)
                self.pushButton_10.setEnabled(False)
                self.pushButton_11.setEnabled(False)
                self.pushButton_12.setEnabled(False)
                self.pushButton_14.setEnabled(False)
            else:
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(False)
                self.pushButton_4.setEnabled(False)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_7.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)
                self.pushButton_10.setEnabled(True)
                self.pushButton_11.setEnabled(True)
                self.pushButton_12.setEnabled(False)
                self.pushButton_14.setEnabled(False)

    def over(self):
        global end
        global set_speed
        global tracking
        tracking = False
        QMessageBox.about(self, "비디오 끝", "마지막 프레임입니다 초기화해주세요")
        end = True
        set_speed = 1
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.pushButton_11.setEnabled(False)
        self.pushButton_12.setEnabled(False)

    # def vidload(self):
    #     stream = cv2.VideoCapture(video_path[0])
    #     # count2 = 0
    #     if flush:
    #         return
    #
    #     while True:
    #         if not Q.full():
    #             # count2 += 1
    #             # print("working? %d" % count2)
    #             ret, frame = stream.read()
    #             original_image = frame
    #
    #             image_data = image_preprocess(np.copy(original_image), [input_size, input_size])  # 인풋 프레임 전처리
    #             image_data = tf.expand_dims(image_data, 0)
    #
    #             if not ret:
    #                 return
    #             Q.put((ret, image_data, original_image))
    #
    #         elif Q.full():
    #             time.sleep(0.005)

    def speed_up(self):
        self.pushButton_12.setEnabled(True)
        global set_speed
        set_speed += 1
        self.label7.setText("배속  x%d 배" % set_speed)

    def speed_down(self):
        global set_speed
        if set_speed == 2:
            set_speed = 1
            self.label7.setText("배속  x%d 배" % set_speed)
            self.pushButton_12.setEnabled(False)
            return
        set_speed -= 1
        self.label7.setText("배속  x%d 배" % set_speed)

    def open_folder(self):
        if not os.path.isdir('./captured/'):
            QMessageBox.about(self, "폴더 없음", "captured 폴더가 생성되지 않았습니다")
        else:
            # subprocess.Popen(['xdg-open', './captured/'])
            os.system('xdg-open "%s"' % './captured/')
        return

    def target_only_view(self):
        global target_only_view
        if not target_only_view:
            target_only_view = True
            self.label2.setPixmap(QPixmap.fromImage(qimg_1))
        else:
            target_only_view = False
            self.label2.setPixmap(QPixmap.fromImage(qimg_2))

    def track(self):
        tracker.tracks = []
        tracker._next_id = 1  # 트래커초기화

        self.pushButton_4.setEnabled(False)
        vid = cv2.VideoCapture(video_path[0])
        Track_only = ['person']
        global framecount
        framecount = 0
        times = []
        global handler
        global qimg_1, qimg_2
        global tracking
        tracking = True

        # if os.path.isfile('./captured/obj%d/%d.txt' % (copied_text, copied_text)):
        #     with open('./captured/obj%d/%d.txt' % (copied_text, copied_text), 'w') as f:
        #         f.write("")

        while True:
            if not os.path.isdir('./captured/obj%d' % copied_text):
                os.mkdir('./captured/obj%d' % copied_text)

            myobject = text
            ret, img = vid.read()

            if set_speed > 1:
                for i in range(set_speed - 1):
                    ret, img = vid.read()
                    framecount += 1

                while pause:
                    self.pushButton_5.setEnabled(False)
                    self.pushButton_6.setEnabled(False)
                    self.pushButton_8.setEnabled(False)
                    time.sleep(0.005)
                    if flush:
                        return
                    if not pause:
                        self.pushButton_5.setEnabled(True)
                        self.pushButton_6.setEnabled(True)
                        self.pushButton_8.setEnabled(True)
                        break
            else:
                pass

            # ret, image_data, original_image = Q.get()
            if not ret:
                self.over()
                return
            else:
                pass

            while pause:
                self.pushButton_5.setEnabled(False)
                self.pushButton_6.setEnabled(False)
                self.pushButton_8.setEnabled(False)
                time.sleep(0.005)
                if flush:
                    return
                if not pause:
                    self.pushButton_5.setEnabled(True)
                    self.pushButton_6.setEnabled(True)
                    self.pushButton_8.setEnabled(True)
                    break

            original_image = img

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])  # 인풋 프레임 전처리
            image_data = tf.expand_dims(image_data, 0)

            # try:
            #     original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # except:
            #     break

            t1 = time.time()
            pred_bbox = YoloV3.predict(image_data)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')  # 신뢰도 낮은 박스 제거하는 커스텀 함수 from utils

            # extract bboxes to boxes (x, y, width, height), scores and names
            boxes, scores, names = [], [], []
            for bbox in bboxes:
                if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                    boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                                  bbox[3].astype(int) - bbox[1].astype(int)])
                    scores.append(bbox[4])
                    names.append(NUM_CLASS[int(bbox[5])])

            # Obtain all the detections for the given frame.
            boxes = np.array(boxes)
            names = np.array(names)
            scores = np.array(scores)
            features = np.array(encoder(original_image, boxes))
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(boxes, scores, names, features)]

            # Pass detections to the deepsort object and obtain the track information.

            tracker.predict()
            tracker.update(detections)

            t2 = time.time()
            times.append(t2 - t1)
            times = times[-20:]
            fps = int(1000 / (sum(times) / len(times) * 1000))

            # Obtain info from the tracks
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5: # currently tracked objects is in tracker.tracks and its updated time count is time_since_update 5시간단위 이상 넘은것들은 그냥 넘긴
                    continue
                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                class_name = track.get_class()  # Get the class name of particular object
                tracking_id = track.track_id  # Get the ID for the particular track
                index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
                tracked_bboxes.append(bbox.tolist() + [tracking_id,
                                                       index])  # Structure data, that we could use it with our draw_bbox function

            # if myobject is not None and len(tracked_bboxes) != 0:
            if len(tracked_bboxes) != 0:

                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_7.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)
                self.pushButton_10.setEnabled(True)
                self.pushButton_11.setEnabled(True)
                self.pushButton_14.setEnabled(True)

                # print(framecount)

                copied_tracked_bboxes = []
                for i, value in enumerate(tracked_bboxes):
                    if value[4] == myobject:
                        copied_tracked_bboxes = [tracked_bboxes.pop(i)]
                        break
                    else:
                        copied_tracked_bboxes = []
                        pass

                if not copied_tracked_bboxes:
                    global objimg
                    global qimg_1, qimg_2
                    image = cv2.putText(original_image, "Time: {:.1f} FPS \n Tracking Fail".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), 2)
                    h, w, ch = image.shape
                    bytesPerLine = ch * w
                    qimg_1 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
                    qimg_2 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                    if not target_only_view:
                        self.label2.setPixmap(QPixmap.fromImage(qimg_2))
                    else:
                        self.label2.setPixmap(QPixmap.fromImage(qimg_1))
                    objimg = []
                    pass

                else:
                    x1 = int(copied_tracked_bboxes[0][0])
                    y1 = int(copied_tracked_bboxes[0][1])
                    x2 = int(copied_tracked_bboxes[0][2])
                    y2 = int(copied_tracked_bboxes[0][3])
                    objimg = np.array(original_image[y1:y2, x1:x2])

                    image = cv2.putText(original_image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = draw_bbox(image, copied_tracked_bboxes, CLASSES=CLASSES, Text_colors=(255, 255, 255),
                                          rectangle_colors=(0, 128, 0), tracking=True)
                    h, w, ch = image.shape
                    bytesPerLine = ch * w
                    qimg_1 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                    image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
                    qimg_2 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                    if not target_only_view:
                        self.label2.setPixmap(QPixmap.fromImage(qimg_2))
                    else:
                        self.label2.setPixmap(QPixmap.fromImage(qimg_1))

                # times3 = []
                # p1 = time.time()
                #
                # cv2.imwrite("./captured/frame.jpg", image)
                # self.img_load()
                #
                # p2 = time.time()
                # times3.append(p2 - p1)
                # times3 = times3[-20:]
                # fps3 = int(1000 / (sum(times3) / len(times3) * 1000))

                times_2 = []
                t3 = time.time()
                times_2.append(t3 - t1)
                times_2 = times_2[-20:]
                fps2 = int(1000 / (sum(times_2) / len(times_2) * 1000))

                print(framecount, ", fps:", fps2)

            else:
                pass

            framecount += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MyWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
