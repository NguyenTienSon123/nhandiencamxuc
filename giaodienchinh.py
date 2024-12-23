# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QMessageBox
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(677, 484)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Label tiêu đề
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 20, 561, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        # Nút Tải ảnh lên
        self.bt_UL = QtWidgets.QPushButton(self.centralwidget)
        self.bt_UL.setGeometry(QtCore.QRect(130, 100, 93, 28))
        self.bt_UL.setObjectName("bt_UL")

        # Nút Sử dụng webcam
        self.bt_WC = QtWidgets.QPushButton(self.centralwidget)
        self.bt_WC.setGeometry(QtCore.QRect(480, 100, 131, 28))
        self.bt_WC.setObjectName("bt_WC")

        # Hiển thị ảnh gốc
        self.anh_goc = QLabel(self.centralwidget)
        self.anh_goc.setGeometry(QtCore.QRect(30, 150, 361, 271))
        self.anh_goc.setObjectName("anh_goc")

        # Hiển thị khuôn mặt nhận diện
        self.anh_mat = QLabel(self.centralwidget)
        self.anh_mat.setGeometry(QtCore.QRect(480, 170, 151, 111))
        self.anh_mat.setObjectName("anh_mat")

        # Hiển thị cảm xúc
        self.cam_xuc = QLabel(self.centralwidget)
        self.cam_xuc.setGeometry(QtCore.QRect(460, 350, 201, 31))
        self.cam_xuc.setObjectName("cam_xuc")

        # Label "Khuôn mặt"
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(520, 290, 81, 16))
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 677, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Kết nối các nút với các hàm
        self.bt_UL.clicked.connect(self.open_image)
        self.bt_WC.clicked.connect(self.start_webcam)

        # Load model nhận diện cảm xúc
        try:
            self.model = load_model('emotion_model.keras')
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            QMessageBox.critical(None, "Lỗi", f"Lỗi khi tải model hoặc Cascade Classifier: {e}")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Nhận diện cảm xúc từ khuôn mặt"))
        self.label.setText(_translate("MainWindow", "Nhận diện cảm xúc từ khuôn mặt"))
        self.bt_UL.setText(_translate("MainWindow", "Tải ảnh lên"))
        self.bt_WC.setText(_translate("MainWindow", "Sử dụng webcam"))
        self.label_2.setText(_translate("MainWindow", "Khuôn mặt"))

    def open_image(self):
        # Mở file dialog để chọn ảnh
        fname, _ = QFileDialog.getOpenFileName(None, "Chọn ảnh", "", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            # Hiển thị ảnh gốc lên anh_goc
            pixmap = QPixmap(fname)
            self.anh_goc.setPixmap(pixmap.scaled(self.anh_goc.size(), QtCore.Qt.KeepAspectRatio))

            # Đọc ảnh bằng OpenCV và chuyển sang màu xám
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Nhận diện khuôn mặt
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Kiểm tra nếu không có khuôn mặt nào được phát hiện
            if len(faces) == 0:
                self.cam_xuc.setText("Không nhận diện được khuôn mặt")
            else:
                # Xử lý khuôn mặt đầu tiên được phát hiện
                for (x, y, w, h) in faces:
                    face = img[y:y + h, x:x + w]

                    # Hiển thị khuôn mặt lên anh_mat
                    face_qt = self.convert_cv_to_qt(face)
                    self.anh_mat.setPixmap(face_qt.scaled(self.anh_mat.size(), QtCore.Qt.KeepAspectRatio))

                    # Phát hiện cảm xúc
                    roi_gray = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                    roi_gray = roi_gray.astype("float32") / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=0)
                    roi_gray = np.expand_dims(roi_gray, axis=-1)

                    # Dự đoán cảm xúc
                    prediction = self.model.predict(roi_gray)
                    label = self.emotion_labels[np.argmax(prediction)]

                    # Cập nhật nhãn cảm xúc trên giao diện
                    self.cam_xuc.setText(label)
                    break  # Chỉ xử lý khuôn mặt đầu tiên

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Nhận diện khuôn mặt và cảm xúc
            frame = self.detect_face_and_emotion(frame, display=False)
            cv2.imshow('Webcam', frame)
            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_face_and_emotion(self, frame, display=True):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = cv2.resize(gray_frame[y:y + h, x:x + w], (48, 48))
            roi_gray = roi_gray.astype("float32") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Dự đoán cảm xúc
            prediction = self.model.predict(roi_gray)
            label = self.emotion_labels[np.argmax(prediction)]

            # Vẽ khung và hiển thị nhãn
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Hiển thị khuôn mặt và cảm xúc trên giao diện
            if display:
                self.display_image(frame[y:y + h, x:x + w], self.anh_mat)
                self.cam_xuc.setText(label)
        if not display:
            return frame

    def convert_cv_to_qt(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img)

    def display_image(self, img, widget):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        widget.setPixmap(QtGui.QPixmap.fromImage(q_img))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
