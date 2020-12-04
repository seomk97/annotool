# @inproceedings{Wojke2017simple,
#   title={Simple Online and Realtime Tracking with a Deep Association Metric},
#   author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
#   booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
#   year={2017},
#   pages={3645--3649},
#   organization={IEEE},
#   doi={10.1109/ICIP.2017.8296962}
# }
#
# @inproceedings{Wojke2018deep,
#   title={Deep Cosine Metric Learning for Person Re-identification},
#   author={Wojke, Nicolai and Bewley, Alex},
#   booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
#   year={2018},
#   pages={748--756},
#   organization={IEEE},
#   doi={10.1109/WACV.2018.00087}
# }

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
input_object = None
copied_input_object = None
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
escape = 0

w_checked = False
r_checked = False
s_checked = False

YoloV4 = yolo  # yolo : tensorflow weight transformed from darknet weight at main.py
score_threshold = 0.3
iou_threshold = 0.1
CLASSES = YOLO_COCO_CLASSES
max_cosine_distance = 0.4
nn_budget = None

# initialize deep sort object
model_filename = "./pjtlibs/mars-small128.pb"  # deep sort weight
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

NUM_CLASS = read_class_names(CLASSES)  # name strip from utils
key_list = list(NUM_CLASS.keys())
val_list = list(NUM_CLASS.values())


class SignalOfTrack(QObject):
    frameCount = pyqtSignal(int)
    buttonName = pyqtSignal(str, bool)
    pixmapImage = pyqtSignal(QPixmap)

    def slider_run(self, int):
        self.frameCount.emit(int)

    def btn_run(self, str, bool):
        self.buttonName.emit(str, bool)

    def pixmap_run(self, QPixmap):
        self.pixmapImage.emit(QPixmap)


class MainWindow(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_file.clicked.connect(self.file_load)
        self.btn_load.clicked.connect(self.screen_load)
        self.btn_object.clicked.connect(self.object_select)
        self.btn_track.clicked.connect(self.my_thread)
        self.btn_reset.clicked.connect(self.q_key)
        self.btn_play.clicked.connect(self.space_key)
        self.btn_target.clicked.connect(self.target_change)
        self.btn_up.clicked.connect(self.speed_up)
        self.btn_down.clicked.connect(self.speed_down)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_tab.clicked.connect(self.target_only_view)
        self.btn_json.clicked.connect(self.make_json)
        self.btn_delete.clicked.connect(self.item_delete)
        self.btn_action_toggle.clicked.connect(self.record_action_toggle)
        self.horizontalSlider.sliderMoved.connect(self.slider_moved)
        self.horizontalSlider.sliderReleased.connect(self.slider_released)
        self.horizontalSlider.sliderPressed.connect(self.slider_pressed)
        self.listWidget.itemDoubleClicked.connect(self.item_double_clicked)
        self.actionQuit.triggered.connect(qApp.quit)
        self.actionQuit.setShortcut('Ctrl+Q')
        self.btn_file.setShortcut('f')
        self.btn_load.setShortcut('l')
        self.btn_object.setShortcut('o')
        self.btn_track.setShortcut('t')
        self.btn_reset.setShortcut('q')
        self.btn_play.setShortcut(Qt.Key.Key_Space)
        self.btn_target.setShortcut('c')
        self.btn_up.setShortcut(Qt.Key.Key_Right)
        self.btn_down.setShortcut(Qt.Key.Key_Left)
        self.btn_folder.setShortcut(Qt.Key.Key_Home)
        self.btn_tab.setShortcut(Qt.Key.Key_Tab)
        self.btn_json.setShortcut('j')
        self.btn_delete.setShortcut(Qt.Key.Key_Delete)
        self.btn_action_toggle.setShortcut('b')

    @pyqtSlot(QPixmap)
    def pixmap_update(self, QPixmap):
        self.label_mainscreen.setPixmap(QPixmap)

    @pyqtSlot(int)
    def slider_control(self, int):
        self.horizontalSlider.setValue(int)

    @pyqtSlot(str, bool)
    def btn_control(self, str, bool):
        if str == 'btn_action_toggle':
            self.btn_action_toggle.setEnabled(bool)
        elif str == 'btn_delete':
            self.btn_delete.setEnabled(bool)
        elif str == 'btn_down':
            self.btn_down.setEnabled(bool)
        elif str == 'btn_file':
            self.btn_file.setEnabled(bool)
        elif str == 'btn_folder':
            self.btn_folder.setEnabled(bool)
        elif str == 'btn_json':
            self.btn_json.setEnabled(bool)
        elif str == 'btn_load':
            self.btn_load.setEnabled(bool)
        elif str == 'btn_object':
            self.btn_object.setEnabled(bool)
        elif str == 'btn_play':
            self.btn_play.setEnabled(bool)
        elif str == 'btn_reset':
            self.btn_reset.setEnabled(bool)
        elif str == 'btn_tab':
            self.btn_tab.setEnabled(bool)
        elif str == 'btn_target':
            self.btn_target.setEnabled(bool)
        elif str == 'btn_track':
            self.btn_track.setEnabled(bool)
        elif str == 'btn_up':
            self.btn_up.setEnabled(bool)
        else:
            raise Exception('btn invalid')

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
            self.label_end_frame.setText('%d' % num_of_frame)
            temp_vid.release()
        else:
            pass
        if not video_path:
            self.btn_load.setEnabled(False)
        else:
            if video_path_buffer[0] == '':
                return
            else:
                self.btn_load.setEnabled(True)
                self.btn_reset.setEnabled(True)
                return

    def img_load(self):
        pixmap = QPixmap("./captured/frame.jpg")
        self.label_mainscreen.setPixmap(pixmap)
        return

    def screen_load(self):
        global flush
        global pause
        flush = False
        pause = False
        Object_tracking(yolo, video_path[0], '', input_size=input_size, show=True, iou_threshold=0.3,
                        rectangle_colors=(255, 0, 0), Track_only=["person"])
        self.img_load()
        if os.path.isfile("./captured/frame.jpg"):
            os.remove("./captured/frame.jpg")
        self.btn_object.setEnabled(True)
        self.btn_load.setEnabled(False)
        self.btn_reset.setEnabled(True)
        return

    def object_select(self):
        global input_object
        global copied_input_object
        global pause
        global writing_dir
        global workspace

        if not tracking:
            input_object, ok = QInputDialog.getInt(self, 'Object Select', 'Please input Object number')
            copied_input_object = input_object
            if ok:
                self.label_object.setText('obj ' + str(copied_input_object))
                self.label_target.setText('person ' + str(input_object))
                self.btn_track.setEnabled(True)
            else:
                input_object = None
                self.label_object.setText("None")
                self.label_target.setText("None")
                self.btn_track.setEnabled(False)
                return

        if tracking:
            if pause:
                pass
            else:
                self.space_key()

            input_object_2, ok = QInputDialog.getInt(self, 'Object Select', 'Please input Object number')
            if ok:
                if input_object != input_object_2:
                    self.make_json()
                    writing_dir = ""
                    workspace = []
                    self.listWidget.clear()
                else:
                    pass
                input_object = input_object_2
                copied_input_object = input_object
                self.label_object.setText('obj ' + str(input_object))
                self.label_target.setText('person ' + str(input_object))
                self.space_key()
            else:
                return

    def target_change(self):
        global input_object
        global pause
        global target_changed

        if pause:
            pass
        else:
            self.space_key()

        input_object_2, ok = QInputDialog.getInt(self, 'Changing Target', 'Please input desired target number')
        if ok:
            input_object = input_object_2
            self.label_target.setText('person ' + str(input_object))
            target_changed = 1  # target_change state flag
            return
        else:
            return

    def my_thread(self):
        self.centralwidget.setFocus()
        self.btn_file.setEnabled(False)
        self.btn_track.setChecked(True)
        self.btn_track.setEnabled(False)
        self.btn_play.setChecked(True)
        self.btn_play.setText('Pause\n(space)')
        self.btn_play.setShortcut(Qt.Key.Key_Space)
        self.btn_target.setEnabled(True)
        self.btn_up.setEnabled(True)
        self.btn_tab.setEnabled(True)
        self.btn_action_toggle.setEnabled(True)
        self.horizontalSlider.setEnabled(False)
        th = threading.Thread(target=self.track)
        th.setDaemon(True)
        th.start()

    def w_key(self):
        global workspace
        global action_started
        label_n_count = [copied_input_object, framecount]

        if button_checkable:
            if w_checked:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   start_walking" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return
            else:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   end_walking" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return

        else:
            if objimg.size == 0:
                self.label_show_label.setText("Track Failed")
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

            self.label_show_label.setText("%d.jpg   walking" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
            self.label_show_target.setPixmap(pixmap_small)
            return

    def r_key(self):
        global workspace
        global action_started
        label_n_count = [copied_input_object, framecount]

        if button_checkable:
            if r_checked:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   start_running" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return
            else:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   end_running" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return

        else:
            if objimg.size == 0:
                self.label_show_label.setText("Track Failed")
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

            self.label_show_label.setText("%d.jpg   running" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
            self.label_show_target.setPixmap(pixmap_small)
            return

    def s_key(self):
        global workspace
        global action_started
        label_n_count = [copied_input_object, framecount]

        if button_checkable:
            if s_checked:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   start_stop" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return
            else:
                if objimg.size == 0:
                    self.label_show_label.setText("Track Failed")
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

                self.label_show_label.setText("%d.jpg   end_stop" % label_n_count[1])
                pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
                self.label_show_target.setPixmap(pixmap_small)
                return

        else:
            if objimg.size == 0:
                self.label_show_label.setText("Track Failed")
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

            self.label_show_label.setText("%d.jpg   stop" % label_n_count[1])
            pixmap_small = QPixmap(writing_dir + "/%d.jpg" % label_n_count[1])
            self.label_show_target.setPixmap(pixmap_small)
            return

    def space_key(self):
        global pause
        if not pause:
            self.horizontalSlider.setEnabled(True)
            pause = True
            self.btn_play.setChecked(False)
            self.btn_play.setText('Play\n(space)')
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            self.centralwidget.setFocus()
        else:
            self.horizontalSlider.setEnabled(False)
            pause = False
            self.btn_play.setChecked(True)
            self.btn_play.setText('Pause\n(space)')
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            self.btn_tab.setEnabled(True)
        return

    def q_key(self):
        self.flush()
        return

    def flush(self):
        global input_object
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

        reply = QMessageBox.question(self, 'Reset', 'Do you want to proceed?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            tracking = False
            flush = True
            set_speed = 1
            self.label.setText("File Path")
            self.label_object.setText("None")
            self.label_target.setText("None")
            self.label_speed.setText("speed  x%d " % set_speed)
            self.label_show_label.setText("")
            if os.path.isfile("./captured/frame.jpg"):
                os.remove("./captured/frame.jpg")
            self.img_load()
            pixmap = QPixmap("./captured/frame.jpg")
            self.label_show_target.setPixmap(pixmap)
            self.btn_file.setEnabled(True)
            self.btn_load.setEnabled(False)
            self.btn_object.setEnabled(False)
            self.btn_track.setChecked(False)
            self.btn_track.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.btn_play.setChecked(False)
            self.btn_play.setEnabled(False)
            self.btn_target.setEnabled(False)
            self.btn_up.setEnabled(False)
            self.btn_down.setEnabled(False)
            self.btn_tab.setEnabled(False)
            self.btn_action_toggle.setEnabled(False)
            self.make_json()
            input_object = None
            end = False
            self.horizontalSlider.setValue(1)
            self.horizontalSlider.setEnabled(False)
            self.listWidget.clear()
            workspace = []
            writing_dir = ""
            return
        else:
            if end:
                self.btn_reset.setEnabled(True)

                return
            else:
                return

    def video_end(self):
        global end
        global set_speed
        self.horizontalSlider.setValue(self.horizontalSlider.maximum())
        end = True
        set_speed = 1
        self.label_speed.setText("speed  x%d " % set_speed)
        self.btn_reset.setEnabled(True)
        # QMessageBox.about(self, "Video ended", "This is the last frame")  # focus issue don't use
        return

    def speed_up(self):
        self.btn_down.setEnabled(True)
        global set_speed
        set_speed += 1
        self.label_speed.setText("speed  x%d " % set_speed)
        return

    def speed_down(self):
        global set_speed
        if set_speed == 2:
            set_speed = 1
            self.label_speed.setText("speed  x%d " % set_speed)
            self.btn_down.setEnabled(False)
            return
        set_speed -= 1
        self.label_speed.setText("speed  x%d " % set_speed)
        return

    def open_folder(self):
        if not os.path.isdir('./captured/'):
            QMessageBox.about(self, "Couldn't find directory", "Please generate ../captured/")
        else:
            os.system('xdg-open "%s"' % './captured/')
        return

    def target_only_view(self):
        global target_only_view
        if not target_only_view:
            target_only_view = True
            self.label_mainscreen.setPixmap(QPixmap.fromImage(qimg_1))

            return
        else:
            target_only_view = False
            self.label_mainscreen.setPixmap(QPixmap.fromImage(qimg_2))
            return

    def slider_pressed(self):
        self.btn_tab.setEnabled(False)

    def slider_moved(self):
        global slider_moved, jump_to_frame, end
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()
        end = False

    def slider_released(self):
        global slider_moved, jump_to_frame, escape, objimg
        # self.centralwidget.setFocus()
        slider_moved = True
        jump_to_frame = self.horizontalSlider.value()
        escape = 1
        objimg = np.array([])

    def item_double_clicked(self):
        global jumped, jump_to_frame, end
        if pause:
            self.space_key()
        else:
            pass
        item_index = self.listWidget.currentRow()
        self.horizontalSlider.setValue(workspace[item_index][0])
        print(self.horizontalSlider.sliderPosition())
        jump_to_frame = self.horizontalSlider.value()
        pixmap_small = QPixmap(writing_dir + "/%d.jpg" % workspace[item_index][0])
        self.label_show_target.setPixmap(pixmap_small)
        self.label_show_label.setText("%d.jpg   %s" % (workspace[item_index][0], workspace[item_index][1]))
        jumped = True
        end = False
        return

    def make_json(self):
        workspace_frame_list = []
        workspace_label_list = []
        if workspace:
            for i, workspace_things in enumerate(workspace):
                workspace_frame_list.append(int(workspace_things[0]))
                workspace_label_list.append(workspace_things[1])
            if not pause:
                self.space_key()
            QMessageBox.about(self, "Save complete", "Saved at  %s " % writing_dir[:])
        else:
            return

        json_dict = {workspace_frame_list[i]: workspace_label_list[i] for i in range(len(workspace_frame_list))}
        with open(writing_dir + "/%d.json" % copied_input_object, "w") as json_file:
            json.dump(json_dict, json_file)
        return

    def item_delete(self):
        global workspace
        if self.listWidget.selectedItems():
            item = workspace.pop(self.listWidget.currentRow())
            os.remove(writing_dir + "/%d.jpg" % item[0])
            self.listWidget.takeItem(self.listWidget.currentRow())
            dummy_pixmap = QPixmap(writing_dir + "/%d.jpg" % item[0])
            self.label_show_target.setPixmap(dummy_pixmap)
            self.label_show_label.setText("")
            return
        else:
            return

    def record_action_toggle(self):
        global button_checkable
        global toggle_button
        global workspace
        global w_checked, r_checked, s_checked

        toggle_button = self.btn_action_toggle.isChecked()
        if toggle_button:
            button_checkable = True
            self.btn_action_toggle.setText("Action End (B)")
            self.btn_action_toggle.setShortcut('b')
            return
        else:
            self.btn_action_toggle.setText("Action Start (B)")
            self.btn_action_toggle.setShortcut('b')
            if w_checked:
                w_checked = False
                self.w_key()

            elif r_checked:
                r_checked = False
                self.r_key()

            elif s_checked:
                s_checked = False
                self.s_key()

            button_checkable = False
            return

    def keyPressEvent(self, e):
        global w_checked, r_checked, s_checked
        if e.key() == Qt.Key_W:
            if button_checkable:
                if not w_checked and not r_checked and not s_checked:
                    w_checked = True
                else:
                    return
                self.w_key()
            else:
                self.w_key()
        elif e.key() == Qt.Key_R:
            if button_checkable:
                if not w_checked and not r_checked and not s_checked:
                    r_checked = True
                else:
                    return
                self.r_key()
            else:
                self.r_key()
        elif e.key() == Qt.Key_S:
            if button_checkable:
                if not w_checked and not r_checked and not s_checked:
                    s_checked = True
                else:
                    return
                self.s_key()
            else:
                self.s_key()

    def track(self):
        signal = SignalOfTrack()
        signal.frameCount.connect(self.slider_control)
        signal.buttonName.connect(self.btn_control)
        signal.pixmapImage.connect(self.pixmap_update)

        tracker.tracks = []
        tracker._next_id = 1  # init tracker

        vid = cv2.VideoCapture(video_path[0])

        Track_only = ['person']
        global framecount, pause_flag, qimg_1, qimg_2, tracking, slider_moved, objimg, jumped, target_changed, pause, writing_dir, set_speed, token, escape

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
        token = 0
        ret = 0
        img = 0
        while True:

            t1 = time.time()

            if not os.path.isdir('./captured/obj%d' % copied_input_object):
                writing_dir = "./captured/obj%d" % copied_input_object
                os.mkdir(writing_dir)
            else:
                i = 1
                while writing_dir == "":  # object changed while tracking, writing_dir becomes "" and perform loop once
                    if os.path.isdir('./captured/obj%d_%d' % (copied_input_object, i)):
                        i += 1
                        continue
                    else:
                        writing_dir = "./captured/obj%d_%d" % (copied_input_object, i)
                        os.mkdir(writing_dir)
                        break

            myobject = input_object

            if pause:
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                pass
            else:
                ret, img = vid.read()
                if not ret:  # video end event
                    self.space_key()
                    self.video_end()
                else:
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

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
                signal.btn_run('btn_play', False)
                signal.btn_run('btn_action_toggle', False)
                signal.btn_run('btn_tab', False)
                signal.btn_run('btn_object', False)
                signal.btn_run('btn_target', False)
                pause_flag = 1
                pass
            elif jump_count > 8:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                self.space_key()
                signal.btn_run('btn_play', True)
                signal.btn_run('btn_action_toggle', True)
                signal.btn_run('btn_tab', True)
                signal.btn_run('btn_object', True)
                signal.btn_run('btn_target', True)
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
                            if ret:
                                h, w, ch = img.shape
                                bytesPerLine = ch * w
                                qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                                signal.pixmap_run(QPixmap.fromImage(qimg_3))
                                framecount = jump_to_frame
                                slider_moved = False
                            else:
                                pass
                        else:
                            pass

                        signal.slider_run(framecount)

                        if not copied_tracked_bboxes:
                            signal.btn_run('btn_action_toggle', False)

                        if target_changed or jumped or pause_flag:
                            break
                        if flush:
                            return
                        if not pause:
                            token = 1
                            break

                    if target_changed or escape:
                        if escape:
                            escape = 0
                        pass
                    else:
                        for i in range(set_speed - 1):
                            ret, img = vid.read()
                            signal.slider_run(vid.get(cv2.CAP_PROP_POS_FRAMES))
                            framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

            else:
                pass

            while pause:
                if slider_moved:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
                    tracker.tracks = []
                    tracker._next_id = 1
                    ret, img = vid.read()
                    if ret:
                        h, w, ch = img.shape
                        bytesPerLine = ch * w
                        qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                        signal.pixmap_run(QPixmap.fromImage(qimg_3))
                        framecount = jump_to_frame
                        slider_moved = False
                    else:
                        pass
                else:
                    pass

                signal.slider_run(framecount)

                if not copied_tracked_bboxes:
                    signal.btn_run('btn_action_toggle', False)

                if escape:
                    escape = 0
                    break
                if target_changed or jumped or pause_flag:
                    break
                if flush:
                    return
                if not pause:
                    token = 1
                    break

            if token:
                token = 0
                continue

            if target_changed == 1:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                ret, img = vid.read()
                myobject = input_object
                target_changed = 0

            if jumped:
                continue

            original_image = img
            signal.slider_run(framecount)

            if not ret:
                if not pause:
                    self.space_key()
                self.video_end()
                continue
            else:
                pass

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = tf.expand_dims(image_data, 0)

            pred_bbox = YoloV4.predict(image_data)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')  # nms from utils

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
                if not track.is_confirmed() or track.time_since_update > 1:  # currently tracked objects is in tracker.tracks and its updated time count is time_since_update
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

            if jump_count is None:
                signal.btn_run('btn_reset', True)
                signal.btn_run('btn_play', True)
                signal.btn_run('btn_target', True)
                signal.btn_run('btn_up', True)
                signal.btn_run('btn_action_toggle', True)
            else:
                pass

            copied_tracked_bboxes = []
            for i, value in enumerate(tracked_bboxes):
                if value[4] == myobject:
                    copied_tracked_bboxes = [tracked_bboxes.pop(i)]
                    break
                else:
                    copied_tracked_bboxes = []
                    pass

            if not copied_tracked_bboxes:
                signal.btn_run('btn_action_toggle', False)

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
                    signal.pixmap_run(QPixmap.fromImage(qimg_2))
                else:
                    signal.pixmap_run(QPixmap.fromImage(qimg_1))

                objimg = np.array([])
                pass

            else:
                x1 = int(copied_tracked_bboxes[0][0])
                y1 = int(copied_tracked_bboxes[0][1])
                x2 = int(copied_tracked_bboxes[0][2])
                y2 = int(copied_tracked_bboxes[0][3])

                box_margin = 17

                objimg = np.array(original_image[y1-box_margin:y2+box_margin, x1-box_margin:x2+box_margin])  # target image size to be saved

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
                    signal.pixmap_run(QPixmap.fromImage(qimg_2))
                else:
                    signal.pixmap_run(QPixmap.fromImage(qimg_1))

            fps2 = int(fps)
            print(framecount, ", fps:", fps2)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MainWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
