from __future__ import with_statement

# Load gui file
import os

from gui import Ui_MainWindow

import sys
from PyQt5 import QtGui, QtWidgets
import numpy as np
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
import tensorflow as tf
import copy
from pandas import read_csv

# load model weights
GLOBAL_model_DenseNet = tf.keras.models.load_model('Models/DenseNetV1.hdf5')
GLOBAL_model_EffNet_Best = tf.keras.models.load_model('Models/Best_acc_EffNet.h5')
GLOBAL_model_EffNet_Normal = tf.keras.models.load_model('Models/Old_Normal_EffNet.h5')
### GLOBAL_model_EffNet_Old = tf.keras.models.load_model('EffNetV1.h5')
GLOBAL_model_EffNet_Optimized = tf.keras.models.load_model('Models/EffNetOptimized.h5')
# create results folder
path = r'results'

# df = read_csv(r'C:\Users\Mohamed Bushnaq\PycharmProjects\Biometrics\Cephalo\train_senior.csv')
df = read_csv(r'True labels\train_senior.csv')
true_labels = df.values[:, 1:]

if not os.path.exists(path):
    os.makedirs(path)
BoxIdx = 0
Test_images = np.arange(1, 401, 1)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # connect ui listeners to functions
        self.ui.actionUpload.triggered.connect(lambda: openfile())
        self.ui.actionSave.triggered.connect(lambda: save_image())
        self.ui.model_1_btn.clicked.connect(lambda: load_AI_MODEL(copy.deepcopy(img_dict["img"]), 0))
        self.ui.model_2_btn.clicked.connect(lambda: load_AI_MODEL(copy.deepcopy(img_dict["img"]), 1))
        self.ui.model_3_btn_effOpt.clicked.connect(lambda: load_AI_MODEL(copy.deepcopy(img_dict["img"]), 2))
        self.ui.checkBox.clicked.connect(lambda: true_landmarks())

        # self.ui.model_3_btn_effOpt.clicked.connect(lambda: show_MSE_error())

        """
        Model index 0 for Efficinetnet B2
        Model index 1 for Densenet
        Model index 2 for Latest EffNet
        """

        model_names = ["EffNet_2Fusion", "DenseNet", "EffNet_3Fusion"]

        img_dict = {
            "path": 0,
            "orig_img": 0,
            "img": 0,
            "pts": np.empty([38, ]),
            "outputDense": 0,
            "outputRes": 0
        }

        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("There is no true labels here!")
        msg.setIcon(QMessageBox.Critical)
        msg.setStandardButtons(QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)

        msg_save = QMessageBox()
        msg_save.setWindowTitle("Saving completed")
        msg_save.setText("Your image with landmarks has been saved")
        msg_save.setDetailedText("Please navigate to the results folder that has been created automatically and you "
                                 "will find the save image")
        msg_save.setIcon(QMessageBox.Information)
        msg_save.setStandardButtons(QMessageBox.Ok)
        msg_save.setDefaultButton(QMessageBox.Ok)

        def show_MSE_error(predicted_points_for_error):
            try:
                idx = int(img_dict['path'].split('/')[-1].split('.')[0]) - 1
            except:
                return
                # idx = int(img_dict['path'].split('/')[-1].split('.')[0]) - 1
            if ((idx + 1) not in Test_images):
                # msg.exec_()
                return
            true_labels_for_error = np.array(true_labels[idx] / 10)
            MAE_error = np.mean(np.abs(true_labels_for_error - predicted_points_for_error))
            print(MAE_error)

        def show_true_labels():
            try:
                idx = int(img_dict['path'].split('/')[-1].split('.')[0]) - 1
            except:
                return
            # idx = int(img_dict['path'].split('/')[-1].split('.')[0]) - 1
            example_label = true_labels[idx] / 10
            img = copy.deepcopy(self.output_img)
            for i in range(19):
                x_coor = int(example_label[2 * i + 0])
                y_coor = int(example_label[2 * i + 1])
                img = cv2.circle(img, (int(x_coor), int(y_coor)), 2, (50, 168, 131), -1)
            self.ui.output_img_widget.clear()
            QT_image = QImage(img, img.shape[1], img.shape[0], img.strides[0],
                              QImage.Format_RGB888)
            # self.ui.output_title.setText(model_names[idx] + " Output")
            self.ui.output_img_widget.setPixmap(QtGui.QPixmap.fromImage(QT_image))

        def true_landmarks():
            if BoxIdx == 0:
                return
            try:
                test_img_flag = int(img_dict['path'].split('/')[-1].split('.')[0])
            except:
                if self.ui.checkBox.isChecked():
                    msg.exec_()
                return
            # test_img_flag = int(img_dict['path'].split('/')[-1].split('.')[0])
            if (test_img_flag not in Test_images) and (self.ui.checkBox.isChecked()):
                msg.exec_()
                return

            if (self.ui.checkBox.isChecked()):
                show_true_labels()

            else:
                self.ui.output_img_widget.clear()
                QT_image = QImage(self.output_img, self.output_img.shape[1], self.output_img.shape[0],
                                  self.output_img.strides[0],
                                  QImage.Format_RGB888)
                self.ui.output_title.setText(model_names[self.model_idx] + " Output")
                self.ui.output_img_widget.setPixmap(QtGui.QPixmap.fromImage(QT_image))

        def readImg(path):
            """
            1- read image from file path
            2- convert from BGR to RGB format
            3- resize image to 193x240 to fit model input
            """
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, (193, 240)), img

        def openfile():
            """
            1- open image
            2- set image to the location in the gui
            """
            self.ui.output_img_widget.clear()
            filename = QFileDialog.getOpenFileName()[0]
            img, img_dict['orig_img'] = readImg(filename)
            img_dict["path"] = filename
            img_dict["img"] = img
            Qimage = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            self.ui.original_img_widget.setPixmap(QtGui.QPixmap.fromImage(Qimage))

        def img_LandMarks_adding(img, np_predicted_pts, size=2):
            """
            :param img: Image to embed landmarks to
            :param np_predicted_pts: output of model (x,y) tuple , 19 landmark cord pairs

            plot Landmarks coordinates on the image

            blue for EffNet

            orange for Densenet
            """
            for i in range(19):
                x_coor = int(np_predicted_pts[2 * i + 0])
                y_coor = int(np_predicted_pts[2 * i + 1])
                if self.model_idx == 0:
                    img = cv2.circle(img, (int(x_coor), int(y_coor)), size, (0, 130, 255), -1)
                elif self.model_idx == 1:
                    img = cv2.circle(img, (int(x_coor), int(y_coor)), size, (255, 127, 80), -1)
                else:
                    img = cv2.circle(img, (int(x_coor), int(y_coor)), size, (255, 0, 0), -1)
            return img

        def getDenseNet(input_img):
            return GLOBAL_model_DenseNet.predict(input_img)

        def getEffNet_fusion_bet_3(input_img):
            bst_pred = GLOBAL_model_EffNet_Best.predict(input_img)
            nor_pred = GLOBAL_model_EffNet_Normal.predict(input_img)
            opt_pred = GLOBAL_model_EffNet_Optimized.predict(input_img)
            # dns_pred = GLOBAL_model_DenseNet.predict(input_img)
            # avg_pred = (bst_pred + nor_pred) / 2
            fus_pred = (bst_pred + opt_pred + nor_pred) / 3
            # return avg_pred
            return fus_pred

        def getEffNet_fusion_bet_2(input_img):
            bst_pred = GLOBAL_model_EffNet_Best.predict(input_img)
            opt_pred = GLOBAL_model_EffNet_Optimized.predict(input_img)
            avg_pred = (bst_pred + opt_pred) / 2
            return avg_pred

        # def getDenseNet(input_img):
        #     return GLOBAL_model_EffNet_Old.predict(s)
        # def getDenseNet(input_img):
        #     return GLOBAL_model_EffNet_Best.predict(input_img)
        # def getDenseNet(input_img):
        #     return GLOBAL_model_EffNet_Normal.predict(input_img)

        def load_AI_MODEL(img, idx):
            """
            :param idx:  0 for EffNet model, 1 for DenseNet model
            1- update the
            2- reset the output image widget
            3- get cords from model
            4- places landmarks on the image
            5- display output image
            """
            global BoxIdx  # These two lines for the checkbox issue
            BoxIdx = 1
            self.model_idx = idx
            self.ui.output_img_widget.clear()
            np_img = np.array([img])
            np_img = np_img[0, :, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            np_img = clahe.apply(np_img)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            np_img = tf.expand_dims(np_img, axis=0)
            # np_img = cv2.equalizeHist(np_img)
            if idx == 0:
                predicted_pts = getEffNet_fusion_bet_3(np_img)
            elif idx == 1:
                predicted_pts = getDenseNet(np_img)
            else:
                predicted_pts = getEffNet_fusion_bet_2(np_img)
            np_predicted_pts = np.array([predicted_pts])
            np_predicted_pts = np.squeeze(np_predicted_pts)
            show_MSE_error(np_predicted_pts)
            img_dict['pts'] = np_predicted_pts
            # self.predicted_points_for_error == np_predicted_pts
            opt_img = img_LandMarks_adding(img, np_predicted_pts)
            self.output_img = opt_img
            QT_image = QImage(opt_img, opt_img.shape[1], opt_img.shape[0], opt_img.strides[0],
                              QImage.Format_RGB888)
            self.ui.output_title.setText(model_names[idx] + " Output")
            self.ui.output_img_widget.setPixmap(QtGui.QPixmap.fromImage(QT_image))

        def save_image():
            img_name = img_dict['path'].split('/')[-1].split('.')[0]
            result_name = 'results/' + img_name + "_" + model_names[self.model_idx] + "_output.png"
            opt_img = img_LandMarks_adding(copy.deepcopy(img_dict['orig_img']), np.multiply(img_dict["pts"], 10), 10)
            output = QImage(opt_img, opt_img.shape[1], opt_img.shape[0], opt_img.strides[0],
                            QImage.Format_RGB888)
            output.save(result_name)
            msg_save.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    style = "style.stylesheet"
    with open(style, "r") as f:
        app.setStyleSheet(f.read())
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
