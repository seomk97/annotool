import os
import sys
import threading
import time

import cv2
import numpy as np

# Import the PyTorch backend before PyQt on Windows to avoid c10.dll load issues.
from tracking import create_tracking_preview, draw_tracks, tracker
from camera import FISHEYE_CALIB, configure_fisheye, fisheye, preprocess_frame
from storage import AnnotationSession

from PyQt5 import uic
from PyQt5.QtCore import QObject, QRect, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QShortcut,
    QWidget,
    qApp,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_FPS = os.environ.get("ANNOTOOL_DEBUG_FPS", "").strip().lower() in {"1", "true", "yes", "on"}
UI_PATH = os.path.join(APP_DIR, "assets", "annotool.ui")
form_class = uic.loadUiType(UI_PATH)[0]

video_path = []
input_object = None          # current tracker ID shown on the video
object_name = ""             # user-defined persistent object name
framecount = 0
end = False
stop_requested = False
pause = False
objimg = np.array([])
set_speed = 1.0
target_only_view = False
qimg_1 = QImage()
qimg_2 = QImage()
tracking = False
slider_preview_pending = False
slider_commit_pending = False
slider_dragging = False
jump_to_frame = 0
workspace = []
jumped = False
target_changed = 0
escape = 0

class TrackingSignals(QObject):
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


class AnnotationWindow(QMainWindow, form_class):
    previewReady = pyqtSignal(bool, str)
    previewProgress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.active_action = None
        self.annotation_session = AnnotationSession()

        # Scale the designer geometry from its 1301x730 logical canvas.
        # Avoid reading the central widget size before the first layout pass.
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
        self._last_fisheye_calib = FISHEYE_CALIB

        self.previewReady.connect(self._screen_load_finished)
        self.previewProgress.connect(self._update_load_progress)
        self.btn_file.clicked.connect(self.file_load)
        self.btn_load.clicked.connect(self.screen_load)
        self.btn_object.clicked.connect(self.object_select)
        self.btn_track.clicked.connect(self.toggle_tracking)
        self.btn_reset.clicked.connect(self.reset_session)
        self.btn_target.clicked.connect(self.target_change)
        self.btn_up.clicked.connect(self.speed_up)
        self.btn_down.clicked.connect(self.speed_down)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_tab.clicked.connect(self.target_only_view)
        self.btn_delete.clicked.connect(self.item_delete)
        self.btn_action_toggle.clicked.connect(self.record_action_toggle)
        self.btn_action_snapshot.clicked.connect(self.record_action_snapshot)
        self.horizontalSlider.sliderMoved.connect(self.slider_moved)
        self.horizontalSlider.sliderReleased.connect(self.slider_released)
        self.horizontalSlider.sliderPressed.connect(self.slider_pressed)
        self.listWidget.itemDoubleClicked.connect(self.item_double_clicked)
        self.actionQuit.triggered.connect(qApp.quit)
        self.actionQuit.setShortcut(QKeySequence("Ctrl+Q"))

        # Keep runtime shortcuts on the window so activation is independent
        # of whichever control currently owns keyboard focus.
        for button in (
            self.btn_file,
            self.btn_load,
            self.btn_object,
            self.btn_track,
            self.btn_reset,
            self.btn_target,
            self.btn_up,
            self.btn_down,
            self.btn_folder,
            self.btn_tab,
            self.btn_delete,
            self.btn_action_toggle,
            self.btn_action_snapshot,
        ):
            button.setShortcut(QKeySequence())

        self.btn_file.setText("File (F)")
        self.btn_load.setText("Load (L)")
        self.btn_object.setText("Object (O)")
        self.btn_track.setText("Start Tracking\n(Space)")
        self.btn_reset.setText("Reset (Q)")
        self.btn_target.setText("Box No. (C)")
        self.btn_action_toggle.setText("Action Start (B)")
        self.btn_action_snapshot.setText("Action Snapshot (N)")

        self.btn_up.setEnabled(True)
        self.btn_down.setEnabled(True)

        self._shortcuts = []
        self._add_shortcut("F", lambda: self._click_if_enabled(self.btn_file))
        self._add_shortcut("L", lambda: self._click_if_enabled(self.btn_load))
        self._add_shortcut("O", lambda: self._click_if_enabled(self.btn_object))
        self._add_shortcut(
            QKeySequence(Qt.Key_Space),
            lambda: self._click_if_enabled(self.btn_track),
        )
        self._add_shortcut("Q", lambda: self._click_if_enabled(self.btn_reset))
        self._add_shortcut("C", lambda: self._click_if_enabled(self.btn_target))
        self._add_shortcut(
            QKeySequence(Qt.Key_Right),
            lambda: self._click_if_enabled(self.btn_up),
        )
        self._add_shortcut(
            QKeySequence(Qt.Key_Left),
            lambda: self._click_if_enabled(self.btn_down),
        )
        self._add_shortcut(
            QKeySequence(Qt.Key_Home),
            lambda: self._click_if_enabled(self.btn_folder),
        )
        self._add_shortcut(
            QKeySequence(Qt.Key_Tab),
            lambda: self._click_if_enabled(self.btn_tab),
        )
        self._add_shortcut(
            QKeySequence(Qt.Key_Delete),
            lambda: self._click_if_enabled(self.btn_delete),
        )
        self._add_shortcut(
            "B",
            lambda: self._click_if_enabled(self.btn_action_toggle),
        )
        self._add_shortcut(
            "N",
            lambda: self._click_if_enabled(self.btn_action_snapshot),
        )



    def _add_shortcut(self, sequence, callback):
        shortcut = QShortcut(
            sequence if isinstance(sequence, QKeySequence) else QKeySequence(sequence),
            self,
        )
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    @staticmethod
    def _click_if_enabled(button):
        if button.isEnabled() and button.isVisible():
            button.click()

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
    def slider_control(self, value):
        self.horizontalSlider.setValue(value)

    @pyqtSlot(str, bool)
    def btn_control(self, name, enabled):
        buttons = {
            "btn_action_toggle": self.btn_action_toggle,
            "btn_action_snapshot": self.btn_action_snapshot,
            "btn_delete": self.btn_delete,
            "btn_down": self.btn_down,
            "btn_file": self.btn_file,
            "btn_folder": self.btn_folder,
            "btn_load": self.btn_load,
            "btn_object": self.btn_object,
            "btn_reset": self.btn_reset,
            "btn_tab": self.btn_tab,
            "btn_target": self.btn_target,
            "btn_track": self.btn_track,
            "btn_up": self.btn_up,
        }
        button = buttons.get(name)
        if button is None:
            raise ValueError(f"Unknown button signal: {name}")
        button.setEnabled(enabled)

    def file_load(self):
        global video_path

        video_path_buffer = QFileDialog.getOpenFileName(
            self,
            None,
            None,
            "Video files (*.mp4 *.avi *.mov *.mkv)",
        )
        if video_path_buffer[0] == "":
            return

        default_mode = 1 if fisheye.enabled else 0
        camera_mode, ok = QInputDialog.getItem(
            self,
            "Camera Mode",
            "Input camera:",
            ["Normal", "Fisheye"],
            default_mode,
            False,
        )
        if not ok:
            return

        if camera_mode == "Fisheye":
            start_path = self._last_fisheye_calib or ""
            calibration_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select fisheye calibration",
                start_path,
                "NumPy calibration (*.npz)",
            )
            if not calibration_path:
                return

            self._last_fisheye_calib = calibration_path
            configure_fisheye(calibration_path)
        else:
            configure_fisheye("")

        video_path = video_path_buffer
        self.annotation_session.set_video(video_path[0])
        self.label.setText(video_path[0])

        # Opening a video and showing it must not wait for YOLO/tracker setup.
        # Decode just one raw frame first and display it immediately.
        temp_vid = cv2.VideoCapture(video_path[0])
        if not temp_vid.isOpened():
            QMessageBox.critical(self, "Video open failed", video_path[0])
            return

        num_of_frame = int(temp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = temp_vid.read()
        temp_vid.release()

        if not ret or first_frame is None:
            QMessageBox.critical(self, "Video read failed", "Could not decode the first frame.")
            return

        try:
            first_frame = preprocess_frame(first_frame)
        except Exception as exc:
            QMessageBox.critical(self, "Fisheye calibration failed", str(exc))
            return

        h, w, ch = first_frame.shape
        qimg = QImage(
            first_frame.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888,
        ).rgbSwapped().copy()
        self._set_main_image(qimg)

        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(max(0, num_of_frame))
        self.horizontalSlider.setValue(0)
        self.label_end_frame.setText(str(num_of_frame))

        # Load overlays detection / tracking IDs on the first frame.
        self.btn_load.setText("Load (L)")
        self.btn_load.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_object.setEnabled(False)

        # Only after the raw video is visible do we spend resources preparing
        # the tracking backend.
        if not tracker.is_prepared and not self._model_prepare_started:
            self._model_prepare_started = True
            threading.Thread(
                target=self._prepare_model_background,
                daemon=True,
            ).start()

        return

    def _prepare_model_background(self):
        try:
            tracker.prepare()
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
        global stop_requested
        global pause
        stop_requested = False
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
            tracker.prepare(progress=self.previewProgress.emit)

            self.previewProgress.emit("Reading and tracking first frame...")
            create_tracking_preview(video_path[0])

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

    def _refresh_object_annotations(self):
        global workspace

        workspace = self.annotation_session.annotations_for_object(object_name)
        self.listWidget.clear()
        for record in workspace:
            self.listWidget.addItem(
                f" {record['frame_number']}    {record['action']} "
            )

    def _reset_active_action(self):
        self.active_action = None
        self.btn_action_toggle.setChecked(False)
        self.btn_action_toggle.setText("Action Start (B)")
        self.btn_action_toggle.setToolTip("")

    def object_select(self):
        global input_object
        global object_name
        global pause

        was_playing = tracking and not pause
        if was_playing:
            self.space_key()

        name, ok = QInputDialog.getText(
            self,
            "Object Name",
            "Object name:",
            text=object_name,
        )
        name = name.strip()
        if not ok or not name:
            if was_playing and pause:
                self.space_key()
            return

        target_id, ok = QInputDialog.getInt(
            self,
            "Box No.",
            "Current box number:",
            value=input_object if input_object is not None else 1,
            min=0,
        )
        if not ok:
            if was_playing and pause:
                self.space_key()
            return

        if object_name and name != object_name:
            self._reset_active_action()

        object_name = name
        input_object = target_id

        self.label_object.setText(object_name)
        self.label_target.setText(f"person {input_object}")
        self._refresh_object_annotations()
        self.btn_track.setEnabled(True)
        self.btn_track.setText("Start Tracking\n(Space)" if not tracking else "Resume\n(Space)")

        if was_playing and pause:
            self.space_key()

    def target_change(self):
        global input_object
        global pause
        global target_changed

        was_playing = tracking and not pause
        if was_playing:
            self.space_key()

        input_object_2, ok = QInputDialog.getInt(
            self,
            "Box No.",
            "Current box number:",
            value=input_object if input_object is not None else 1,
            min=0,
        )
        if ok:
            input_object = input_object_2
            self.label_target.setText(f"person {input_object}")
            target_changed = 1

        if was_playing and pause:
            self.space_key()
        return

    def toggle_tracking(self):
        if not self.btn_track.isEnabled():
            return

        if not tracking:
            self._start_tracking_worker()
        else:
            self.space_key()

    def _start_tracking_worker(self):
        global tracking

        tracking = True
        self.centralwidget.setFocus()
        self.btn_file.setEnabled(False)
        self.btn_track.setEnabled(True)
        self.btn_track.setText("Pause\n(Space)")
        self.btn_reset.setEnabled(True)
        self.btn_target.setEnabled(True)
        self.btn_up.setEnabled(True)
        self.btn_down.setEnabled(True)
        self.btn_tab.setEnabled(True)
        self.btn_action_toggle.setEnabled(True)
        self.horizontalSlider.setEnabled(False)
        th = threading.Thread(target=self.track, daemon=True)
        th.start()

    def space_key(self):
        global pause
        pause = not pause
        self._apply_pause_ui(pause)
        return

    @pyqtSlot(bool)
    def _apply_pause_ui(self, paused):
        if paused:
            self.horizontalSlider.setEnabled(True)
            self.btn_track.setText("Resume\n(Space)")
            self.centralwidget.setFocus()
        else:
            self.horizontalSlider.setEnabled(False)
            self.btn_track.setText("Pause\n(Space)")
            self.btn_tab.setEnabled(True)

    def reset_session(self):
        self._reset_state()

    def _reset_state(self):
        global input_object
        global end
        global stop_requested
        global pause
        global set_speed
        global tracking
        global workspace
        global object_name

        if pause:
            pass
        else:
            self.space_key()

        reply = QMessageBox.question(self, 'Reset', 'Do you want to proceed?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            tracking = False
            stop_requested = True
            set_speed = 1.0
            self._reset_active_action()
            self.label.setText("File Path")
            self.label_object.setText("None")
            self.label_target.setText("None")
            object_name = ""
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
            self.btn_track.setEnabled(False)
            self.btn_track.setText("Start Tracking\n(Space)")
            self.btn_reset.setEnabled(False)
            self.btn_target.setEnabled(False)
            self.btn_up.setEnabled(True)
            self.btn_down.setEnabled(set_speed > 0.1)
            self.btn_tab.setEnabled(False)
            self.btn_action_toggle.setEnabled(False)
            self.btn_action_snapshot.setEnabled(False)
            input_object = None
            end = False
            self.horizontalSlider.setValue(1)
            self.horizontalSlider.setEnabled(False)
            self.listWidget.clear()
            workspace = []
            self.annotation_session.clear()
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
        path = self.annotation_session.output_path()
        os.makedirs(path, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system('open "%s"' % path)
        else:
            os.system('xdg-open "%s"' % path)

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
        global slider_dragging
        slider_dragging = True
        self.btn_tab.setEnabled(False)

    def slider_moved(self):
        global slider_preview_pending, jump_to_frame, end
        slider_preview_pending = True
        jump_to_frame = self.horizontalSlider.value()
        end = False

    def slider_released(self):
        global slider_preview_pending, slider_commit_pending
        global slider_dragging, jump_to_frame, escape, objimg

        # Preview and release are separate requests. A worker finishing an
        # earlier preview cannot erase this final boxed-seek request.
        slider_dragging = False
        slider_preview_pending = False
        slider_commit_pending = True
        jump_to_frame = self.horizontalSlider.value()
        escape = 0
        objimg = np.array([])

    def item_double_clicked(self):
        global jumped, jump_to_frame, end

        if pause:
            self.space_key()

        row = self.listWidget.currentRow()
        if row < 0 or row >= len(workspace):
            return

        record = workspace[row]
        frame = int(record["frame_number"])
        self.horizontalSlider.setValue(frame)
        jump_to_frame = frame

        image_path = self.annotation_session.image_path(
            record["object_name"],
            frame,
        )
        self.label_show_target.setPixmap(QPixmap(image_path))
        self.label_show_label.setText(
            f"{os.path.basename(image_path)}   {record['action']}"
        )
        jumped = True
        end = False

    def item_delete(self):
        global workspace

        row = self.listWidget.currentRow()
        if row < 0 or row >= len(workspace):
            return

        record = workspace[row]
        self.annotation_session.delete_annotation(
            record["object_name"],
            int(record["frame_number"]),
        )
        self._refresh_object_annotations()
        self.label_show_target.clear()
        self.label_show_label.setText("")

    def _record_action_marker(self, label):
        global workspace

        if objimg.size == 0:
            self.label_show_label.setText("Track Failed")
            return False
        if not object_name:
            self.label_show_label.setText("No Object")
            return False

        current_frame = int(framecount)
        try:
            image_path = self.annotation_session.save_annotation(
                object_name=object_name,
                action=label,
                frame_number=current_frame,
                image=objimg,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False

        self._refresh_object_annotations()
        row = next(
            (
                index
                for index, record in enumerate(workspace)
                if int(record["frame_number"]) == current_frame
            ),
            -1,
        )
        if row >= 0:
            self.listWidget.setCurrentRow(row)
            if row == len(workspace) - 1:
                self.listWidget.scrollToBottom()

        self.label_show_label.setText(
            f"{os.path.basename(image_path)}   {label}"
        )
        self.label_show_target.setPixmap(QPixmap(image_path))
        return True

    def record_action_snapshot(self):
        global pause

        was_playing = not pause
        if was_playing:
            self.space_key()

        try:
            action, ok = QInputDialog.getText(
                self,
                "Action Snapshot",
                "Action name:",
            )
            action = action.strip()
            if not ok or not action:
                return

            self._record_action_marker(action)
        finally:
            if was_playing and pause:
                self.space_key()

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
                self.btn_action_toggle.setText("Action End (B)")
                self.btn_action_toggle.setToolTip(action)
                return

            if self.active_action is None:
                self.btn_action_toggle.setText("Action Start (B)")
                return

            if not self._record_action_marker(f"end_{self.active_action}"):
                # Tracking may be temporarily lost. Keep the action open so the
                # user can end it on a later valid frame.
                self.btn_action_toggle.setChecked(True)
                self.btn_action_toggle.setText("Action End (B)")
                return

            self.active_action = None
            self.btn_action_toggle.setText("Action Start (B)")
            self.btn_action_toggle.setToolTip("")
            return
        finally:
            if was_playing and pause:
                self.space_key()

    def track(self):
        signal = TrackingSignals()
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

        global framecount, pause_flag, qimg_1, qimg_2, tracking, slider_preview_pending, slider_commit_pending, slider_dragging, objimg, jumped, target_changed, pause, set_speed, token, escape

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

        def render_paused_seek(target_frame):
            global framecount, qimg_1, qimg_2, objimg

            target_frame = max(0, int(target_frame))
            vid.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            tracker.reset()

            ok, seek_image = vid.read()
            if not ok or seek_image is None:
                return

            seek_image = preprocess_frame(seek_image)
            framecount = target_frame
            tracked = tracker.track_frame(seek_image)

            target_box = None
            other_boxes = []
            for box in tracked:
                if box[4] == input_object and target_box is None:
                    target_box = box
                else:
                    other_boxes.append(box)

            base = seek_image.copy()
            if target_box is not None:
                x1, y1, x2, y2 = [int(v) for v in target_box[:4]]
                margin = 17
                h_img, w_img = seek_image.shape[:2]
                x1c = max(0, x1 - margin)
                y1c = max(0, y1 - margin)
                x2c = min(w_img, x2 + margin)
                y2c = min(h_img, y2 + margin)
                objimg = seek_image[y1c:y2c, x1c:x2c].copy()

                target_only = draw_tracks(
                    base.copy(),
                    [target_box],
                    text_color=(255, 255, 255),
                    rectangle_color=(0, 128, 0),
                )
                all_boxes = draw_tracks(target_only.copy(), other_boxes)
                signal.btn_run('btn_action_toggle', True)
                signal.btn_run('btn_action_snapshot', True)
            else:
                objimg = np.array([])
                target_only = base.copy()
                all_boxes = draw_tracks(base.copy(), tracked)
                signal.btn_run('btn_action_toggle', False)
                signal.btn_run('btn_action_snapshot', False)

            h_img, w_img, ch = target_only.shape
            bytes_per_line = ch * w_img
            qimg_1 = QImage(
                target_only.data,
                w_img,
                h_img,
                bytes_per_line,
                QImage.Format_RGB888,
            ).rgbSwapped().copy()
            qimg_2 = QImage(
                all_boxes.data,
                w_img,
                h_img,
                bytes_per_line,
                QImage.Format_RGB888,
            ).rgbSwapped().copy()

            signal.pixmap_run(qimg_1 if target_only_view else qimg_2)
            signal.slider_run(target_frame)
            signal.btn_run('btn_tab', True)

        while True:

            loop_started = time.perf_counter()
            t1 = time.time()

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
                signal.btn_run('btn_track', False)
                signal.btn_run('btn_action_toggle', False)
                signal.btn_run('btn_action_snapshot', False)
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
                signal.btn_run('btn_track', True)
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
                        if slider_commit_pending:
                            seek_target = jump_to_frame
                            slider_commit_pending = False
                            slider_preview_pending = False
                            render_paused_seek(seek_target)
                        elif slider_preview_pending:
                            preview_target = jump_to_frame
                            slider_preview_pending = False
                            vid.set(cv2.CAP_PROP_POS_FRAMES, preview_target)
                            ret, img = vid.read()
                            if ret:
                                img = preprocess_frame(img)
                                h, w, ch = img.shape
                                bytesPerLine = ch * w
                                qimg_3 = QImage(
                                    img.data,
                                    w,
                                    h,
                                    bytesPerLine,
                                    QImage.Format_RGB888,
                                ).rgbSwapped().copy()
                                signal.pixmap_run(qimg_3)
                                framecount = preview_target

                        signal.slider_run(framecount)
                        time.sleep(0.02)

                        if target_changed or jumped or pause_flag:
                            break
                        if stop_requested:
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
                if slider_commit_pending:
                    seek_target = jump_to_frame
                    slider_commit_pending = False
                    slider_preview_pending = False
                    render_paused_seek(seek_target)
                elif slider_preview_pending:
                    preview_target = jump_to_frame
                    slider_preview_pending = False
                    vid.set(cv2.CAP_PROP_POS_FRAMES, preview_target)
                    ret, img = vid.read()
                    if ret:
                        img = preprocess_frame(img)
                        h, w, ch = img.shape
                        bytesPerLine = ch * w
                        qimg_3 = QImage(
                            img.data,
                            w,
                            h,
                            bytesPerLine,
                            QImage.Format_RGB888,
                        ).rgbSwapped().copy()
                        signal.pixmap_run(qimg_3)
                        framecount = preview_target

                signal.slider_run(framecount)
                time.sleep(0.02)

                if escape:
                    escape = 0
                    break
                if target_changed or jumped or pause_flag:
                    break
                if stop_requested:
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

            original_image = preprocess_frame(img) if ret else img
            signal.slider_run(framecount)

            if not ret:
                if not pause:
                    pause = True
                    signal.pause_run(True)
                signal.video_end_run()
                continue

            # Run person detection, ReID and multi-object association off the GUI thread.
            tracked_bboxes = tracker.track_frame(original_image)
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
                signal.btn_run('btn_action_snapshot', action_enabled)
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
                image = draw_tracks(image, tracked_bboxes)
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
                h_img, w_img = original_image.shape[:2]
                x1c = max(0, x1 - box_margin)
                y1c = max(0, y1 - box_margin)
                x2c = min(w_img, x2 + box_margin)
                y2c = min(h_img, y2 + box_margin)
                objimg = original_image[y1c:y2c, x1c:x2c].copy()

                image = cv2.putText(original_image, " {:.1f} FPS".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), 2)
                image = cv2.putText(image, " Tracking Success", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 128, 0), 2)
                image = cv2.putText(image, " %d frame" % vid.get(cv2.CAP_PROP_POS_FRAMES), (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 0), 2)
                image = draw_tracks(
                    image,
                    copied_tracked_bboxes,
                    text_color=(255, 255, 255),
                    rectangle_color=(0, 128, 0),
                )
                h, w, ch = image.shape
                bytesPerLine = ch * w
                qimg_1 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()
                image = draw_tracks(image, tracked_bboxes)
                qimg_2 = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped().copy()

                if not target_only_view:
                    signal.pixmap_run(qimg_2)
                else:
                    signal.pixmap_run(qimg_1)

            if DEBUG_FPS:
                print(framecount, ", fps:", int(fps))

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


def run_app():
    application = QApplication(sys.argv)
    window = AnnotationWindow()
    window.show()
    return application.exec_()
