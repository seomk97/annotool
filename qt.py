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
import threading
import json

# Import the PyTorch/Ultralytics backend before PyQt on Windows.
# PyTorch can fail to load c10.dll when imported after Qt DLLs.
from main import *

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

form_class = uic.loadUiType("./pjtlibs/qtui.ui")[0]

video_path = []
input_object = None
copied_input_object = None
framecount = 0
end = False
flush = False
pause = False
objimg = np.array([])
set_speed = 1.0
target_only_view = False
qimg_1 = QImage()
qimg_2 = QImage()
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

score_threshold = 0.05
iou_threshold = 0.70
CLASSES = YOLO_COCO_CLASSES

# The YOLO26s + TrackTrack ReID adapter is imported from main.py as `tracker`.
# Keep detector/tracker state outside the Qt button/signal state machine.

class SignalOfTrack(QObject):
    frameCount = pyqtSignal(int)
    buttonName = pyqtSignal(str, bool)
    pixmapImage = pyqtSignal(QImage)
    pauseState = pyqtSignal(bool)
    videoEnded = pyqtSignal()

    def slider_run(self, value):
        self.frameCount.emit(int(value))

    def btn_run(self, name, enabled):
        self.buttonName.emit(name, enabled)

    def pixmap_run(self, image):
        self.pixmapImage.emit(image)

    def pause_run(self, paused):
        self.pauseState.emit(paused)

    def video_end_run(self):
        self.videoEnded.emit()


class MainWindow(QMainWindow, form_class):
    previewReady = pyqtSignal(bool, str)
    previewProgress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Custom action recording state.
        self.active_action = None

        # Preserve the original 1301x751 visual layout, but scale widget
        # geometries when the user resizes the main window.
        # qtui.ui was designed for a 1301x751 main window with a 21 px
        # menu bar, so the central widget's logical design area is 1301x730.
        # Do not read centralwidget.size() here: before the first show/layout
        # pass Qt may still report a temporary tiny size.
        self._base_central_size = QSize(1301, 730)
        self._base_geometries = {
            child: QRect(child.geometry())
            for child in self.centralwidget.children()
            if isinstance(child, QWidget)
        }
        self.setMinimumSize(900, 520)
        self.resize(1301, 751)
        self.label_mainscreen.setScaledContents(False)
        self.label_mainscreen.setAlignment(Qt.AlignCenter)
        self._current_main_image = QImage()
        self._load_dialog = None
        self._model_prepare_started = False
        self._model_prepare_error = None

        self.previewReady.connect(self._screen_load_finished)
        self.previewProgress.connect(self._update_load_progress)
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

        # Start the expensive detector/tracker warm-up as soon as the window
        # exists. In normal use it runs while the user is choosing a video, so
        # the first Load click usually needs only first-frame inference.
        if not yolo.is_prepared and not self._model_prepare_started:
            self._model_prepare_started = True
            threading.Thread(
                target=self._prepare_model_background,
                daemon=True,
            ).start()

    @pyqtSlot(QImage)
    def pixmap_update(self, image):
        self._set_main_image(image)

    def _set_main_image(self, image):
        if image is None or image.isNull():
            self._current_main_image = QImage()
            self.label_mainscreen.clear()
            return

        self._current_main_image = image.copy()
        self._render_main_image()

    def _render_main_image(self):
        if self._current_main_image.isNull():
            return

        pixmap = QPixmap.fromImage(self._current_main_image)
        pixmap = pixmap.scaled(
            self.label_mainscreen.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label_mainscreen.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if not hasattr(self, "_base_geometries"):
            return

        base_w = max(1, self._base_central_size.width())
        base_h = max(1, self._base_central_size.height())
        scale_x = self.centralwidget.width() / base_w
        scale_y = self.centralwidget.height() / base_h

        for widget, rect in self._base_geometries.items():
            widget.setGeometry(
                int(rect.x() * scale_x),
                int(rect.y() * scale_y),
                max(1, int(rect.width() * scale_x)),
                max(1, int(rect.height() * scale_y)),
            )

        self._render_main_image()

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
        if video_path_buffer[0] != '' and video_path_buffer != video_path:
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

    def _prepare_model_background(self):
        try:
            yolo.prepare()
        except Exception as exc:
            self._model_prepare_error = str(exc)
        finally:
            self._model_prepare_started = False

    def img_load(self):
        image = QImage("./captured/frame.jpg")
        if image.isNull():
            self._current_main_image = QImage()
            self.label_mainscreen.clear()
        else:
            self._set_main_image(image)
        return

    def screen_load(self):
        global flush
        global pause
        flush = False
        pause = False

        self.btn_load.setEnabled(False)
        self.btn_object.setEnabled(False)

        self._load_dialog = QProgressDialog(
            "Preparing model...",
            None,
            0,
            0,
            self,
        )
        self._load_dialog.setWindowTitle("Loading video")
        self._load_dialog.setWindowModality(Qt.WindowModal)
        self._load_dialog.setMinimumDuration(0)
        self._load_dialog.setAutoClose(False)
        self._load_dialog.setAutoReset(False)
        self._load_dialog.setCancelButton(None)
        self._load_dialog.show()

        threading.Thread(target=self._screen_load_worker, daemon=True).start()

    @pyqtSlot(str)
    def _update_load_progress(self, message):
        if self._load_dialog is not None:
            self._load_dialog.setLabelText(message)

    def _screen_load_worker(self):
        try:
            self.previewProgress.emit("Loading model / tracker...")
            yolo.prepare(progress=self.previewProgress.emit)

            self.previewProgress.emit("Reading and tracking first frame...")
            Object_tracking(
                yolo,
                video_path[0],
                '',
                input_size=input_size,
                show=True,
                iou_threshold=iou_threshold,
                rectangle_colors=(255, 0, 0),
                Track_only=["person"],
            )

            self.previewProgress.emit("Rendering preview...")
            self.previewReady.emit(True, "")
        except Exception as exc:
            self.previewReady.emit(False, str(exc))

    @pyqtSlot(bool, str)
    def _screen_load_finished(self, ok, error):
        if self._load_dialog is not None:
            self._load_dialog.close()
            self._load_dialog.deleteLater()
            self._load_dialog = None

        if not ok:
            self.btn_load.setEnabled(True)
            QMessageBox.critical(self, "Preview failed", error)
            return

        self.img_load()
        if os.path.isfile("./captured/frame.jpg"):
            os.remove("./captured/frame.jpg")
        self.btn_object.setEnabled(True)
        self.btn_reset.setEnabled(True)

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
        self.btn_play.setEnabled(True)
        self.btn_play.setText('Pause\n(space)')
        self.btn_play.setShortcut(Qt.Key.Key_Space)
        self.btn_reset.setEnabled(True)
        self.btn_target.setEnabled(True)
        self.btn_up.setEnabled(True)
        self.btn_down.setEnabled(True)
        self.btn_tab.setEnabled(True)
        self.btn_action_toggle.setEnabled(True)
        self.horizontalSlider.setEnabled(False)
        th = threading.Thread(target=self.track, daemon=True)
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
        pause = not pause
        self._apply_pause_ui(pause)
        return

    @pyqtSlot(bool)
    def _apply_pause_ui(self, paused):
        if paused:
            self.horizontalSlider.setEnabled(True)
            self.btn_play.setChecked(False)
            self.btn_play.setText('Play\n(space)')
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            self.centralwidget.setFocus()
        else:
            self.horizontalSlider.setEnabled(False)
            self.btn_play.setChecked(True)
            self.btn_play.setText('Pause\n(space)')
            self.btn_play.setShortcut(Qt.Key.Key_Space)
            self.btn_tab.setEnabled(True)

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
            set_speed = 1.0
            self.active_action = None
            self.btn_action_toggle.setChecked(False)
            self.btn_action_toggle.setText("Action Start (B)")
            self.label.setText("File Path")
            self.label_object.setText("None")
            self.label_target.setText("None")
            self._update_speed_label()
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
        set_speed = 1.0
        self._update_speed_label()
        self.btn_reset.setEnabled(True)
        # QMessageBox.about(self, "Video ended", "This is the last frame")  # focus issue don't use
        return

    def _update_speed_label(self):
        self.label_speed.setText(f"speed  x{set_speed:.1f} ")

    def speed_up(self):
        global set_speed
        set_speed = round(set_speed + 0.1, 1)
        self.btn_down.setEnabled(set_speed > 0.1)
        self._update_speed_label()
        return

    def speed_down(self):
        global set_speed
        set_speed = max(0.1, round(set_speed - 0.1, 1))
        self.btn_down.setEnabled(set_speed > 0.1)
        self._update_speed_label()
        return

    def open_folder(self):
        path = os.path.abspath("./captured")
        os.makedirs(path, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system('open "%s"' % path)
        else:
            os.system('xdg-open "%s"' % path)
        return

    def target_only_view(self):
        global target_only_view

        # Before the first tracked frame arrives there is no alternate image
        # to display yet. Ignore the toggle instead of passing an invalid
        # placeholder into QPixmap.fromImage().
        next_image = qimg_1 if not target_only_view else qimg_2
        if next_image.isNull():
            return

        target_only_view = not target_only_view
        self._set_main_image(next_image)
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

    def _record_action_marker(self, label):
        global workspace

        if objimg.size == 0 or not writing_dir:
            self.label_show_label.setText("Track Failed")
            return False

        current_frame = int(framecount)
        image = objimg.copy()
        image_path = os.path.join(writing_dir, f"{current_frame}.jpg")

        if not cv2.imwrite(image_path, image):
            QMessageBox.warning(self, "Save failed", f"Could not save {image_path}")
            return False

        # Keep one annotation per frame, matching the original workspace model.
        for index in range(len(workspace) - 1, -1, -1):
            if int(workspace[index][0]) == current_frame:
                workspace.pop(index)
                self.listWidget.takeItem(index)

        workspace.append([current_frame, label, image])
        workspace.sort(key=lambda item: int(item[0]))

        row = next(
            i for i, item in enumerate(workspace)
            if int(item[0]) == current_frame and item[1] == label
        )
        self.listWidget.insertItem(row, f" {current_frame}    {label} ")
        self.listWidget.setCurrentRow(row)
        if row == len(workspace) - 1:
            self.listWidget.scrollToBottom()

        self.label_show_label.setText(f"{current_frame}.jpg   {label}")
        self.label_show_target.setPixmap(QPixmap(image_path))
        return True

    def record_action_toggle(self):
        global pause

        checked = self.btn_action_toggle.isChecked()
        was_playing = not pause

        # Freeze the exact frame while the user types the action name or ends
        # an action. Resume automatically if playback was running before.
        if was_playing:
            self.space_key()

        try:
            if checked:
                action, ok = QInputDialog.getText(
                    self,
                    "Action name",
                    "Action name:",
                )
                action = action.strip()

                if not ok or not action:
                    self.btn_action_toggle.setChecked(False)
                    self.btn_action_toggle.setText("Action Start (B)")
                    return

                if not self._record_action_marker(f"start_{action}"):
                    self.btn_action_toggle.setChecked(False)
                    self.btn_action_toggle.setText("Action Start (B)")
                    return

                self.active_action = action
                self.btn_action_toggle.setText(f"Action End: {action}\n(B)")
                self.btn_action_toggle.setShortcut('b')
                return

            if self.active_action is None:
                self.btn_action_toggle.setText("Action Start (B)")
                return

            if not self._record_action_marker(f"end_{self.active_action}"):
                # Tracking may be temporarily lost. Keep the action open so the
                # user can end it on a later valid frame.
                self.btn_action_toggle.setChecked(True)
                self.btn_action_toggle.setText(f"Action End: {self.active_action}\n(B)")
                return

            self.active_action = None
            self.btn_action_toggle.setText("Action Start (B)")
            self.btn_action_toggle.setShortcut('b')
            return
        finally:
            if was_playing and pause:
                self.space_key()

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
        signal.pauseState.connect(self._apply_pause_ui)
        signal.videoEnded.connect(self.video_end)

        tracker.reset()

        vid = cv2.VideoCapture(video_path[0])
        source_fps = vid.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 1 or source_fps > 240:
            source_fps = 30.0
        frame_interval = 1.0 / source_fps

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
        last_action_enabled = None
        speed_skip_accumulator = 0.0
        while True:

            loop_started = time.perf_counter()
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
                if ret:
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

            if jumped:
                if jump_to_frame > 8:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame - 8)
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

                tracker.reset()
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
                last_action_enabled = None
                pause_flag = 1
                pass
            elif jump_count > 8:
                vid.set(cv2.CAP_PROP_POS_FRAMES, vid.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)
                pause = not pause
                signal.pause_run(pause)
                signal.btn_run('btn_play', True)
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
                            tracker.reset()
                            ret, img = vid.read()
                            if ret:
                                h, w, ch = img.shape
                                bytesPerLine = ch * w
                                qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()
                                signal.pixmap_run(qimg_3)
                                framecount = jump_to_frame
                                slider_moved = False
                            else:
                                pass
                        else:
                            pass

                        signal.slider_run(framecount)
                        time.sleep(0.02)

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
                        speed_skip_accumulator += set_speed - 1.0
                        frames_to_skip = int(speed_skip_accumulator)
                        speed_skip_accumulator -= frames_to_skip

                        for _ in range(frames_to_skip):
                            ret, img = vid.read()
                            if not ret:
                                break
                            signal.slider_run(vid.get(cv2.CAP_PROP_POS_FRAMES))
                            framecount = vid.get(cv2.CAP_PROP_POS_FRAMES)

            else:
                speed_skip_accumulator = 0.0

            while pause:
                if slider_moved:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
                    tracker.reset()
                    ret, img = vid.read()
                    if ret:
                        h, w, ch = img.shape
                        bytesPerLine = ch * w
                        qimg_3 = QImage(img, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()
                        signal.pixmap_run(qimg_3)
                        framecount = jump_to_frame
                        slider_moved = False
                    else:
                        pass
                else:
                    pass

                signal.slider_run(framecount)
                time.sleep(0.02)

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
                    pause = True
                    signal.pause_run(True)
                signal.video_end_run()
                continue

            # Backend modernization: keep the original worker-thread/UI flow,
            # but delegate detection + ID tracking to YOLO26s + TrackTrack ReID.
            tracked_bboxes = tracker.track_frame(
                original_image,
                conf=score_threshold,
                iou=iou_threshold,
                classes=[0],  # person
            )
            t2 = time.time()
            times.append(t2 - t1)
            times = times[-20:]
            fps = 1000 / (sum(times) / len(times) * 1000)

            copied_tracked_bboxes = []
            for i, value in enumerate(tracked_bboxes):
                if value[4] == myobject:
                    copied_tracked_bboxes = [tracked_bboxes.pop(i)]
                    break
                else:
                    copied_tracked_bboxes = []
                    pass

            action_enabled = bool(copied_tracked_bboxes)
            if action_enabled != last_action_enabled:
                signal.btn_run('btn_action_toggle', action_enabled)
                last_action_enabled = action_enabled

            if not copied_tracked_bboxes:
                image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), 2)
                image = cv2.putText(image, " Tracking Fail", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), 2)
                image = cv2.putText(image, " %d frame" % vid.get(cv2.CAP_PROP_POS_FRAMES), (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 0), 2)
                h, w, ch = image.shape
                bytesPerLine = ch * w
                qimg_1 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()
                image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
                qimg_2 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()

                if not target_only_view:
                    signal.pixmap_run(qimg_2)
                else:
                    signal.pixmap_run(qimg_1)

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
                qimg_1 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()
                image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
                qimg_2 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()

                if not target_only_view:
                    signal.pixmap_run(qimg_2)
                else:
                    signal.pixmap_run(qimg_1)

            fps2 = int(fps)
            print(framecount, ", fps:", fps2)

            # Pace sub-1x speeds by time and use an accumulator above 1x so
            # fractional speeds such as 1.1x and 1.7x advance the source
            # timeline accurately on average.
            if not pause and jump_count is None:
                playback_interval = (
                    frame_interval / set_speed
                    if set_speed < 1.0
                    else frame_interval
                )
                remaining = playback_interval - (time.perf_counter() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Annotation_tool = MainWindow()
    Annotation_tool.show()
    sys.exit(app.exec_())
