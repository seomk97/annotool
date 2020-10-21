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
import json

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
box_of_frame = []
framejump = False
jump_to_frame = 0
workspace = []
tracks_temp = []
jumped = False
target_changed = 0

YoloV3 = yolo
score_threshold = 0.3
iou_threshold = 0.1
CLASSES = YOLO_COCO_CLASSES
max_cosine_distance = 0.4
nn_budget = None

# initialize deep sort object
model_filename = "./pjtlibs/mars-small128.pb"  # deep sort 웨이트
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

NUM_CLASS = read_class_names(CLASSES)  # name strip 하는 커스텀함수 from utils
key_list = list(NUM_CLASS.keys())
val_list = list(NUM_CLASS.values())


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
        self.pushButton_15.clicked.connect(self.make_json)
        self.horizontalSlider.sliderMoved.connect(self.slider_moved)
        self.horizontalSlider.sliderReleased.connect(self.slider_released)
        self.listWidget.itemDoubleClicked.connect(self.item_double_clicked)
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
        self.pushButton_13.setShortcut(Qt.Key.Key_Insert)
        self.pushButton_14.setShortcut(Qt.Key.Key_Tab)

    def file_load(self):
        global video_path
        video_path_buffer = QFileDialog.getOpenFileName(self, None, None, "Video files (*.mp4)")
        if video_path_buffer[0] is not '' and video_path_buffer != video_path:
            video_path = video_path_buffer
            self.label.setText(video_path[0])
            self.img_load()
            temp_vid = cv2.VideoCapture(video_path[0])
            num_of_frame = int(temp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.horizontalSlider.setMinimum(0)
            self.horizontalSlider.setMaximum(num_of_frame)
            self.label9.setText('%d' % num_of_frame)
            temp_vid.release()
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
                return

    def img_load(self):
        pixmap = QPixmap("./captured/frame.jpg")
        self.label2.setPixmap(pixmap)
        return

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
        return

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
                self.space_key()
            else:
                return

    def target_change(self):
        global text
        global pause
        global target_changed

        if pause:
            pass
        else:
            self.space_key()

        text_2, ok = QInputDialog.getInt(self, 'Target Change', '타겟번호을 입력해주세요')
        if ok:
            text = text_2
            self.label5.setText('person ' + str(text))
            target_changed = 1
            return
        else:
            return

    def thread(self):
        self.pushButton.setEnabled(False)
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
        global workspace
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("추적실패")
            return
        if os.path.isfile("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1])):
            os.remove("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    workspace.pop(i)
                    self.listWidget.takeItem(i)
                else:
                    pass
        else:
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)

        workspace.append([label_n_count[1], "walking", objimg])
        workspace.sort()

        for i, workspace_item in enumerate(workspace):
            if workspace_item[0] == label_n_count[1]:
                self.listWidget.insertItem(i, "%d   walking" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
                break

            if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                self.listWidget.insertItem(i, "%d   walking" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
            elif workspace_item[0] < label_n_count[1]:
                pass

        self.label4.setText("%d.jpg   walking" % label_n_count[1])
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def r_key(self):
        global workspace
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("추적실패")
            return
        if os.path.isfile("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1])):
            os.remove("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    workspace.pop(i)
                    self.listWidget.takeItem(i)
                else:
                    pass
        else:
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)

        workspace.append([label_n_count[1], "running", objimg])
        workspace.sort()

        for i, workspace_item in enumerate(workspace):
            if workspace_item[0] == label_n_count[1]:
                self.listWidget.insertItem(i, "%d   running" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
                break

            if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                self.listWidget.insertItem(i, "%d   running" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
            elif workspace_item[0] < label_n_count[1]:
                pass

        self.label4.setText("%d.jpg   running" % label_n_count[1])
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def s_key(self):
        global workspace
        label_n_count = [copied_text, framecount]
        if len(objimg) == 0:
            self.label4.setText("추적실패")
            return
        if os.path.isfile("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1])):
            os.remove("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)
            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    workspace.pop(i)
                    self.listWidget.takeItem(i)
                else:
                    pass
        else:
            cv2.imwrite("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]), objimg)

        workspace.append([label_n_count[1], "stop", objimg])
        workspace.sort()

        for i, workspace_item in enumerate(workspace):
            if workspace_item[0] == label_n_count[1]:
                self.listWidget.insertItem(i, "%d   stop" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
                break

            if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                self.listWidget.insertItem(i, "%d   stop" % label_n_count[1])
                if len(workspace)-1 == i:
                    self.listWidget.scrollToBottom()
            elif workspace_item[0] < label_n_count[1]:
                pass

        self.label4.setText("%d.jpg   stop" % label_n_count[1])
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (label_n_count[0], label_n_count[1]))
        self.label6.setPixmap(pixmap_small)
        return

    def space_key(self):
        global pause
        if not pause:
            pause = True
            self.pushButton_9.setText('Play\n(space)')
            self.pushButton_9.setShortcut(Qt.Key.Key_Space)
            self.horizontalSlider.setEnabled(True)
        else:
            pause = False
            self.pushButton_9.setText('Pause\n(space)')
            self.pushButton_9.setShortcut(Qt.Key.Key_Space)
            self.horizontalSlider.setEnabled(False)
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
        global workspace

        if pause:
            pass
        else:
            self.space_key()

        reply = QMessageBox.question(self, 'Message', '초기화합니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            tracking = False
            flush = True
            set_speed = 1
            self.label.setText("Video Path")
            self.label3.setText("None")
            self.label5.setText("None")
            self.label7.setText("배속  x%d 배" % set_speed)
            self.label4.setText("")
            if os.path.isfile("./captured/frame.jpg"):
                os.remove("./captured/frame.jpg")
            self.img_load()
            pixmap = QPixmap("./captured/frame.jpg")
            self.label6.setPixmap(pixmap)
            self.pushButton.setEnabled(True)
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
            self.horizontalSlider.setValue(0)
            self.listWidget.clear()
            workspace = []
            return
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
                return
            else:
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(False)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_7.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)
                self.pushButton_10.setEnabled(True)
                self.pushButton_11.setEnabled(True)
                self.pushButton_12.setEnabled(False)
                self.pushButton_14.setEnabled(True)
                return

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
        return

    def speed_up(self):
        self.pushButton_12.setEnabled(True)
        global set_speed
        set_speed += 1
        self.label7.setText("배속  x%d 배" % set_speed)
        return

    def speed_down(self):
        global set_speed
        if set_speed == 2:
            set_speed = 1
            self.label7.setText("배속  x%d 배" % set_speed)
            self.pushButton_12.setEnabled(False)
            return
        set_speed -= 1
        self.label7.setText("배속  x%d 배" % set_speed)
        return

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
            return
        else:
            target_only_view = False
            self.label2.setPixmap(QPixmap.fromImage(qimg_2))
            return

    def slider_moved(self):
        if tracking and not pause:
            return
        global framejump, jump_to_frame
        framejump = True
        jump_to_frame = self.horizontalSlider.value()

    def slider_released(self):
        if tracking and not pause:
            return
        global framejump, jump_to_frame, target_changed
        framejump = True
        jump_to_frame = self.horizontalSlider.value()
        target_changed = 1

    def item_double_clicked(self):
        # global tracking
        # tracking = False
        global jumped, jump_to_frame
        if pause:
            self.space_key()
        else:
            pass
        item_index = self.listWidget.currentRow()
        self.horizontalSlider.setValue(workspace[item_index][0])
        jump_to_frame = self.horizontalSlider.value()
        pixmap_small = QPixmap("./captured/obj%d/%d.jpg" % (copied_text, workspace[item_index][0]))
        self.label6.setPixmap(pixmap_small)
        self.label4.setText("%d.jpg   %s" % (workspace[item_index][0], workspace[item_index][1]))
        self.horizontalSlider.setValue(workspace[item_index][0])
        # self.slider()
        jumped = True
        # tracking = True
        return

    def make_json(self):
        workspace_frame_list = []
        workspace_label_list = []
        if workspace:
            for i, workspace_things in enumerate(workspace):
                workspace_frame_list.append(int(workspace_things[0]))
                workspace_label_list.append(workspace_things[1])
        else:
            return

        json_dict = {workspace_frame_list[i]: workspace_label_list[i] for i in range(len(workspace_frame_list))}
        # json_val = json.dumps(json_dict)
        with open("./captured/obj%d/%d.json" % (copied_text, copied_text), "w") as json_file:
            json.dump(json_dict, json_file)

        return

    def track(self):
        tracker.tracks = []
        tracker._next_id = 1  # 트래커초기화

        vid = cv2.VideoCapture(video_path[0])

        Track_only = ['person']
        global framecount
        framecount = 0.0
        times = []
        global handler
        global qimg_1, qimg_2
        global tracking
        tracking = True
        global framejump
        global box_of_frame
        global objimg
        global jumped
        jump_count = None
        global target_changed
        global pause

        while True:
            if not os.path.isdir('./captured/obj%d' % copied_text):
                os.mkdir('./captured/obj%d' % copied_text)

            myobject = text

            if pause:
                pass
            else:
                ret, img = vid.read()
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

            if target_changed == 3:
                target_changed = 0
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                self.space_key()

            if target_changed == 2:
                target_changed = 3
                pass

            if not ret:
                self.over()
                return
            else:
                pass

            if jumped:
                if jump_to_frame > 8:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame - 8)
                    framecount = jump_to_frame - 8
                else:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    framecount = 0

                tracker.tracks = []
                tracker._next_id = 1
                jump_count = 0
                jumped = False

            if jump_count is None:
                pass
            elif jump_count <= 10:
                jump_count += 1
                self.pushButton_5.setEnabled(False)
                self.pushButton_6.setEnabled(False)
                self.pushButton_8.setEnabled(False)
                self.pushButton_9.setEnabled(False)
            elif jump_count > 10:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                self.space_key()
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)
                jump_count = None

            if set_speed > 1:
                if jump_count is not None:
                    pass
                else:
                    while pause:
                        if framejump:
                            vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
                            tracker.tracks = []
                            tracker._next_id = 1
                            ret, img = vid.read()
                            h, w, ch = img.shape
                            bytesPerLine = ch * w
                            qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                            self.label2.setPixmap(QPixmap.fromImage(qimg_3))
                            framecount = jump_to_frame
                            self.pushButton_14.setEnabled(True)
                            framejump = False
                        else:
                            pass

                        if not copied_tracked_bboxes:
                            self.pushButton_5.setEnabled(False)
                            self.pushButton_6.setEnabled(False)
                            self.pushButton_8.setEnabled(False)

                        time.sleep(0.005)
                        if target_changed == 1 or target_changed == 2:
                            self.space_key()
                        if jumped:
                            break
                        if flush:
                            return
                        if not pause:
                            # self.pushButton_5.setEnabled(True)
                            # self.pushButton_6.setEnabled(True)
                            # self.pushButton_8.setEnabled(True)
                            break

                    for i in range(set_speed - 1):
                        ret, img = vid.read()
                        self.horizontalSlider.setValue(vid.get(cv2.CAP_PROP_POS_FRAMES))

            else:
                pass

            while pause:
                if framejump:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
                    tracker.tracks = []
                    tracker._next_id = 1
                    ret, img = vid.read()
                    h, w, ch = img.shape
                    bytesPerLine = ch * w
                    qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    self.label2.setPixmap(QPixmap.fromImage(qimg_3))
                    framecount = jump_to_frame
                    self.pushButton_14.setEnabled(True)
                    framejump = False
                else:
                    pass

                if not copied_tracked_bboxes:
                    self.pushButton_5.setEnabled(False)
                    self.pushButton_6.setEnabled(False)
                    self.pushButton_8.setEnabled(False)

                time.sleep(0.005)
                if target_changed == 1 or target_changed == 2:
                    self.space_key()
                if jumped:
                    break
                if flush:
                    return
                if not pause:
                    # self.pushButton_5.setEnabled(True)
                    # self.pushButton_6.setEnabled(True)
                    # self.pushButton_8.setEnabled(True)
                    break

            if target_changed == 1:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                target_changed = 2
                continue

            if jumped:
                continue

            original_image = img
            self.horizontalSlider.setValue(framecount)

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])  # 인풋 프레임 전처리
            image_data = tf.expand_dims(image_data, 0)

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
            fps = 1000 / (sum(times) / len(times) * 1000)

            # Obtain info from the tracks
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1: # currently tracked objects is in tracker.tracks and its updated time count is time_since_update 5시간단위 이상 넘은것들은 그냥 넘긴
                    continue
                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                class_name = track.get_class()  # Get the class name of particular object
                tracking_id = track.track_id  # Get the ID for the particular track
                index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
                tracked_bboxes.append(bbox.tolist() + [tracking_id,
                                                       index])  # Structure data, that we could use it with our draw_bbox function

            if len(tracked_bboxes) != 0:

                if jump_count is None:
                    self.pushButton_5.setEnabled(True)
                    self.pushButton_6.setEnabled(True)
                    self.pushButton_7.setEnabled(True)
                    self.pushButton_8.setEnabled(True)
                    self.pushButton_9.setEnabled(True)
                    self.pushButton_10.setEnabled(True)
                    self.pushButton_11.setEnabled(True)
                else:
                    pass
                self.pushButton_14.setEnabled(True)

                copied_tracked_bboxes = []
                for i, value in enumerate(tracked_bboxes):
                    if value[4] == myobject:
                        copied_tracked_bboxes = [tracked_bboxes.pop(i)]
                        break
                    else:
                        copied_tracked_bboxes = []
                        pass

                if not copied_tracked_bboxes:
                    self.pushButton_5.setEnabled(False)
                    self.pushButton_6.setEnabled(False)
                    self.pushButton_8.setEnabled(False)

                    image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = cv2.putText(image, " Tracking Fail", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
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
                    objimg = np.array(original_image[y1-5:y2+5, x1-5:x2+5])

                    image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = cv2.putText(image, " Tracking Success", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 128, 0), 2)
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

                times_2 = []
                t3 = time.time()
                times_2.append(t3 - t1)
                times_2 = times_2[-20:]
                fps2 = int(1000 / (sum(times_2) / len(times_2) * 1000))

                print(framecount, ", fps:", fps2)

            else:
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MyWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
