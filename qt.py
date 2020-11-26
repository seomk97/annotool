import sys
import time
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from main import *
import threading
import json

form_class = uic.loadUiType("./pjtlibs/qtui.ui")[0]

video_path = []
text = None
copied_text = None
framecount = 0
end = False
flush = False
pause = False
objimg = np.array([])
set_speed = 1
target_only_view = False
qimg_1 = 0
qimg_2 = 0
tracking = False
slider_moved = False
jump_to_frame = 0
workspace = []
jumped = False
target_changed = 0
writing_dir = ""
button_checkable = False  # w,r,s is_checkable
toggle_button = False  # action record toggle button is checked?
action_started = 0  # action record started frame

YoloV4 = yolo  # yolo : tensorflow weight transformed from darknet weight at main.py
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
        self.pushButton_3.clicked.connect(self.object_select)
        self.pushButton_4.clicked.connect(self.my_thread)
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
        self.pushButton_16.clicked.connect(self.item_delete)
        self.pushButton_17.clicked.connect(self.record_action_toggle)
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
        self.pushButton_13.setShortcut(Qt.Key.Key_Home)
        self.pushButton_14.setShortcut(Qt.Key.Key_Tab)
        self.pushButton_15.setShortcut(Qt.Key.Key_Insert)
        self.pushButton_16.setShortcut(Qt.Key.Key_Delete)
        self.pushButton_17.setShortcut('b')

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
        # if os.path.isdir("./captured/"):
        #     if os.listdir("./captured/"):
        #         QMessageBox.about(self, "디렉토리 존재", "파일이 이미 존재합니다")
        #         # return
        #     else:
        #         pass
        # else:
        #     pass

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

    def object_select(self):
        global text
        global copied_text
        global pause
        global writing_dir
        global workspace

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
                if text != text_2:
                    self.make_json()
                    writing_dir = ""
                    workspace = []
                    self.listWidget.clear()
                else:
                    pass
                text = text_2
                copied_text = text
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

    def my_thread(self):
        self.pushButton.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_10.setEnabled(True)
        self.pushButton_11.setEnabled(True)
        self.pushButton_14.setEnabled(True)
        self.pushButton_17.setEnabled(True)
        self.horizontalSlider.setEnabled(False)
        th = threading.Thread(target=self.track)
        th.setDaemon(True)
        th.start()
        return

    def w_key(self):
        global workspace
        global action_started
        label_n_count = [copied_text, framecount]

        if button_checkable:
            button5_checked = self.pushButton_5.isChecked()
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
            self.pushButton_8.setEnabled(False)
            if button5_checked:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "start_walking", objimg])
                workspace.sort()
                action_started = label_n_count[1]

                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_walking " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                        break

                    if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_walking " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                    elif workspace_item[0] < label_n_count[1]:
                        pass

                self.label4.setText("%d.jpg   start_walking" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                return
            else:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "end_walking", objimg])
                workspace.sort()

                check = True
                while check:
                    for i, workspace_item in enumerate(workspace):
                        if not workspace_item[0] or workspace_item[0] >= label_n_count[1]:
                            self.listWidget.insertItem(i, " %d    end_walking " % label_n_count[1])
                            self.listWidget.setCurrentRow(i)
                            if len(workspace) - 1 == i:
                                self.listWidget.scrollToBottom()
                            check = False
                            break
                        elif action_started >= workspace_item[0]:
                            pass
                        elif workspace_item[0] < label_n_count[1]:
                            os.remove(writing_dir + "/%d.jpg" % workspace_item[0])
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                            workspace.sort()
                            break

                self.label4.setText("%d.jpg   end_walking" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                return

        else:
            if len(objimg) == 0:
                self.label4.setText("추적실패")
                return
            if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        workspace.pop(i)
                        self.listWidget.takeItem(i)
                    else:
                        pass
            else:
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

            workspace.append([label_n_count[1], "walking", objimg])
            workspace.sort()

            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    walking " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                    break

                if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    walking " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                elif workspace_item[0] < label_n_count[1]:
                    pass

            self.label4.setText("%d.jpg   walking" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
            self.label6.setPixmap(pixmap_small)
            return

    def r_key(self):
        global workspace
        global action_started
        label_n_count = [copied_text, framecount]

        if button_checkable:
            button5_checked = self.pushButton_6.isChecked()
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
            self.pushButton_8.setEnabled(False)
            if button5_checked:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "start_running", objimg])
                workspace.sort()
                action_started = label_n_count[1]

                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_running " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                        break

                    if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_running " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                    elif workspace_item[0] < label_n_count[1]:
                        pass

                self.label4.setText("%d.jpg   start_running" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                return
            else:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "end_running", objimg])
                workspace.sort()

                check = True
                while check:
                    for i, workspace_item in enumerate(workspace):
                        if not workspace_item[0] or workspace_item[0] >= label_n_count[1]:
                            self.listWidget.insertItem(i, " %d    end_running " % label_n_count[1])
                            self.listWidget.setCurrentRow(i)
                            if len(workspace) - 1 == i:
                                self.listWidget.scrollToBottom()
                            check = False
                            break
                        elif action_started >= workspace_item[0]:
                            pass
                        elif workspace_item[0] < label_n_count[1]:
                            os.remove(writing_dir + "/%d.jpg" % workspace_item[0])
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                            workspace.sort()
                            break

                self.label4.setText("%d.jpg   end_running" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                return

        else:
            if len(objimg) == 0:
                self.label4.setText("추적실패")
                return
            if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        workspace.pop(i)
                        self.listWidget.takeItem(i)
                    else:
                        pass
            else:
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

            workspace.append([label_n_count[1], "running", objimg])
            workspace.sort()

            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    running " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                    break

                if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    running " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                elif workspace_item[0] < label_n_count[1]:
                    pass

            self.label4.setText("%d.jpg   running" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
            self.label6.setPixmap(pixmap_small)
            return

    def s_key(self):
        global workspace
        global action_started
        label_n_count = [copied_text, framecount]

        if button_checkable:
            button5_checked = self.pushButton_8.isChecked()
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
            self.pushButton_8.setEnabled(False)
            if button5_checked:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "start_stop", objimg])
                workspace.sort()
                action_started = label_n_count[1]

                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_stop " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                        break

                    if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                        self.listWidget.insertItem(i, " %d    start_stop " % label_n_count[1])
                        self.listWidget.setCurrentRow(i)
                        if len(workspace) - 1 == i:
                            self.listWidget.scrollToBottom()
                    elif workspace_item[0] < label_n_count[1]:
                        pass

                self.label4.setText("%d.jpg   start_stop" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                return
            else:
                if len(objimg) == 0:
                    self.label4.setText("추적실패")
                    return
                if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                    os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                    for i, workspace_item in enumerate(workspace):
                        if workspace_item[0] == label_n_count[1]:
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                        else:
                            pass
                else:
                    cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

                workspace.append([label_n_count[1], "end_stop", objimg])
                workspace.sort()

                check = True
                while check:
                    for i, workspace_item in enumerate(workspace):
                        if not workspace_item[0] or workspace_item[0] >= label_n_count[1]:
                            self.listWidget.insertItem(i, " %d    end_stop " % label_n_count[1])
                            self.listWidget.setCurrentRow(i)
                            if len(workspace) - 1 == i:
                                self.listWidget.scrollToBottom()
                            check = False
                            break
                        elif action_started >= workspace_item[0]:
                            pass
                        elif workspace_item[0] < label_n_count[1]:
                            os.remove(writing_dir + "/%d.jpg" % workspace_item[0])
                            workspace.pop(i)
                            self.listWidget.takeItem(i)
                            workspace.sort()
                            break

                self.label4.setText("%d.jpg   end_stop" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label6.setPixmap(pixmap_small)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                return

        else:
            if len(objimg) == 0:
                self.label4.setText("추적실패")
                return
            if os.path.isfile(writing_dir + "/%d.jpg" % label_n_count[1]):
                os.remove(writing_dir + "/%d.jpg" % label_n_count[1])
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)
                for i, workspace_item in enumerate(workspace):
                    if workspace_item[0] == label_n_count[1]:
                        workspace.pop(i)
                        self.listWidget.takeItem(i)
                    else:
                        pass
            else:
                cv2.imwrite(writing_dir + "/%d.jpg" % label_n_count[1], objimg)

            workspace.append([label_n_count[1], "stop", objimg])
            workspace.sort()

            for i, workspace_item in enumerate(workspace):
                if workspace_item[0] == label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    stop " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                    break

                if not workspace_item[0] or workspace_item[0] > label_n_count[1]:
                    self.listWidget.insertItem(i, " %d    stop " % label_n_count[1])
                    self.listWidget.setCurrentRow(i)
                    if len(workspace) - 1 == i:
                        self.listWidget.scrollToBottom()
                elif workspace_item[0] < label_n_count[1]:
                    pass

            self.label4.setText("%d.jpg   stop" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
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
        global writing_dir

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
            self.pushButton_17.setEnabled(False)
            self.make_json()
            text = None
            end = False
            self.horizontalSlider.setValue(0)
            self.listWidget.clear()
            workspace = []
            writing_dir = ""
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
                self.pushButton_17.setEnabled(False)
                return
            else:
                return

    def video_end(self):
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
        global slider_moved, jump_to_frame
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()

    def slider_released(self):
        global slider_moved, jump_to_frame, target_changed
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()
        target_changed = 1

    def item_double_clicked(self):
        global jumped, jump_to_frame
        if pause:
            self.space_key()
        else:
            pass
        item_index = self.listWidget.currentRow()
        self.horizontalSlider.setValue(workspace[item_index][0])
        jump_to_frame = self.horizontalSlider.value()
        pixmap_small = QPixmap(writing_dir + "/%d.jpg" % workspace[item_index][0])
        self.label6.setPixmap(pixmap_small)
        self.label4.setText("%d.jpg   %s" % (workspace[item_index][0], workspace[item_index][1]))
        jumped = True
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
        with open(writing_dir + "/%d.json" % copied_text, "w") as json_file:
            json.dump(json_dict, json_file)
        return

    def item_delete(self):
        global workspace
        if self.listWidget.selectedItems():
            item = workspace.pop(self.listWidget.currentRow())
            os.remove(writing_dir + "/%d.jpg" % item[0])
            self.listWidget.takeItem(self.listWidget.currentRow())
            dummy_pixmap = QPixmap(writing_dir + "/%d.jpg" % item[0])
            self.label6.setPixmap(dummy_pixmap)
            self.label4.setText("")
            return
        else:
            return

    def record_action_toggle(self):
        global button_checkable
        global toggle_button
        global workspace

        toggle_button = self.pushButton_17.isChecked()
        if toggle_button:
            button_checkable = True
            self.pushButton_17.setText("Action\nEnd")
            self.pushButton_17.setShortcut('b')
            self.pushButton_5.setCheckable(True)
            self.pushButton_6.setCheckable(True)
            self.pushButton_8.setCheckable(True)
            return
        else:
            button5_checked = self.pushButton_5.isChecked()
            button6_checked = self.pushButton_6.isChecked()
            button8_checked = self.pushButton_8.isChecked()
            self.pushButton_17.setText("Action\nStart")
            self.pushButton_17.setShortcut('b')
            if button5_checked:
                self.pushButton_5.toggle()
                self.w_key()

            elif button6_checked:
                self.pushButton_6.toggle()
                self.r_key()

            elif button8_checked:
                self.pushButton_8.toggle()
                self.s_key()

            self.pushButton_5.setCheckable(False)
            self.pushButton_6.setCheckable(False)
            self.pushButton_8.setCheckable(False)
            button_checkable = False
            return

    def track(self):
        tracker.tracks = []
        tracker._next_id = 1  # 트래커초기화

        vid = cv2.VideoCapture(video_path[0]) # 비디오 불러오기

        Track_only = ['person'] # yolo class중에 person만 bounding box 형성
        global framecount, pause_flag, qimg_1, qimg_2, tracking, slider_moved, objimg, jumped, target_changed, pause, writing_dir

        # framecount = 프레임카운트, pause_flag = 리스트 더블클릭시 이동하고 전프레임 보여주는 루프이후 pause 유지위함
        # pause_flag = temporal pause handler for listwidget item double click loop event
        # qimg_1, qimg_2 = 각각 오리지날 이미지에 대상만 박스처리, 대상만 박스처리한것에 나머지 오브젝트도 박스처리
        # tracking = tracking thread가 돌아가고 있을때 오브젝트 수정을 위한 변수
        # slider_moved = pause 도중 슬라이더가 움직였을 때 메인루프 멈춘상태에서 vid.read로 navigating 용도 bool 변수
        # objimg = 오브젝트 이미지 저장
        # jumped = 리스트 아이템 더블클릭이 된 이벤트 변수
        # target_changed = 타겟변경이 이루어진 이벤트 (기본 0, 변경시 1 전프레임으로 돌아가서 타겟변경후 한번 prediction 후 pause 유지)
        # pause = play/pause event handler
        # writing_dir = object writing directory "./captured/object%d_%d"

        framecount = 0.0
        times = []  # for calculating fps
        tracking = True
        jump_count = None  # count for inner loop (listwidget item double click event)
        pause_flag = 0

        while True:

            t1 = time.time()

            if not os.path.isdir('./captured/obj%d' % copied_text):
                writing_dir = "./captured/obj%d" % copied_text
                os.mkdir(writing_dir)
            else:
                i = 1
                while writing_dir == "":  # object changed while tracking, writing_dir becomes "" and perform loop once
                    if os.path.isdir('./captured/obj%d_%d' % (copied_text, i)):
                        i += 1
                        continue
                    else:
                        writing_dir = "./captured/obj%d_%d" % (copied_text, i)
                        os.mkdir(writing_dir)
                        break

            myobject = text

            if pause:
                pass
            else:
                ret, img = vid.read()
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

            if not ret:  # video end event
                self.video_end()
                return
            else:
                pass

            if jumped:
                if jump_to_frame > 8:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame - 8)
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

                tracker.tracks = []  # tracker initialize
                tracker._next_id = 1  # tracker initialize
                jump_count = 0  # defualt = None, becomes 0 when jumped
                jumped = False

            if jump_count is None:
                pass
            elif jump_count <= 8:
                jump_count += 1
                self.pushButton_5.setEnabled(False)
                self.pushButton_6.setEnabled(False)
                self.pushButton_8.setEnabled(False)
                self.pushButton_9.setEnabled(False)
                self.pushButton_17.setEnabled(False)
                pause_flag = 1
                pass
            elif jump_count > 8:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                self.space_key()
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)
                self.pushButton_17.setEnabled(True)
                jump_count = None
                pause_flag = 0

            if set_speed > 1:
                if jump_count is not None:
                    pass
                else:
                    while pause:
                        if slider_moved:
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
                            slider_moved = False
                        else:
                            pass

                        if not copied_tracked_bboxes:
                            self.pushButton_5.setEnabled(False)
                            self.pushButton_6.setEnabled(False)
                            self.pushButton_8.setEnabled(False)
                            self.pushButton_17.setEnabled(False)

                        time.sleep(0.005)
                        if target_changed == 1:
                            break
                        if jumped:
                            break
                        if pause_flag:
                            break
                        if flush:
                            return
                        if not pause:
                            break

                    if target_changed == 1:
                        pass
                    else:
                        for i in range(set_speed - 1):
                            ret, img = vid.read()
                            self.horizontalSlider.setValue(vid.get(cv2.CAP_PROP_POS_FRAMES))

            else:
                pass

            while pause:
                if slider_moved:
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
                    slider_moved = False
                else:
                    pass

                if not copied_tracked_bboxes:
                    self.pushButton_5.setEnabled(False)
                    self.pushButton_6.setEnabled(False)
                    self.pushButton_8.setEnabled(False)
                    self.pushButton_17.setEnabled(False)

                time.sleep(0.005)
                if target_changed == 1:
                    break
                if jumped:
                    break
                if pause_flag:
                    break
                if flush:
                    return
                if not pause:
                    break

            if target_changed == 1:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                ret, img = vid.read()
                myobject = text
                target_changed = 0
                pass

            if jumped:
                continue

            original_image = img
            self.horizontalSlider.setValue(framecount)

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])  # 인풋 프레임 전처리
            image_data = tf.expand_dims(image_data, 0)

            pred_bbox = YoloV4.predict(image_data)

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

                t2 = time.time()
                times.append(t2 - t1)
                times = times[-20:]
                fps = 1000 / (sum(times) / len(times) * 1000)

            if len(tracked_bboxes) != 0:

                if jump_count is None:
                    if button_checkable:  # 토글키 활성시 사용불가능해야함
                        pass
                    else:
                        self.pushButton_5.setEnabled(True)
                        self.pushButton_6.setEnabled(True)
                        self.pushButton_8.setEnabled(True)

                    self.pushButton_7.setEnabled(True)
                    self.pushButton_9.setEnabled(True)
                    self.pushButton_10.setEnabled(True)
                    self.pushButton_11.setEnabled(True)
                    self.pushButton_17.setEnabled(True)
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
                    self.pushButton_17.setEnabled(False)

                    image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = cv2.putText(image, " Tracking Fail", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = cv2.putText(image, " %d frame" % vid.get(cv2.CAP_PROP_POS_FRAMES), (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 0), 2)
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
                    objimg = np.array(original_image[y1-8:y2+8, x1-8:x2+8])

                    image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), 2)
                    image = cv2.putText(image, " Tracking Success", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 128, 0), 2)
                    image = cv2.putText(image, " %d frame" % vid.get(cv2.CAP_PROP_POS_FRAMES), (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 0), 2)
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

                fps2 = int(fps)
                print(framecount, ", fps:", fps2)

            else:
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MyWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
