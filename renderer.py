
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt

# Add move and zoom functionalities for the image produced.
class GraphicsViewWithZoom(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.initialScale = self.transform().m11()

    def wheelEvent(self, event):
        # Get the current scale factor
        scaleFactor = self.transform().m11()

        # Calculate the new scale factor
        if event.angleDelta().y() > 0:  # Zoom in
            scaleFactor *= 1.25
        else:  # Zoom out
            scaleFactor *= 0.8

        # Set the new scale factor
        self.setTransform(QTransform().scale(scaleFactor, scaleFactor))

    # Reset the scaling of the image.
    def resetScale(self):
        self.setTransform(QTransform().scale(self.initialScale, self.initialScale))

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._mouseX))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._mouseY))
            self._mouseX = event.x()
            self._mouseY = event.y()
        else:
            self._mouseX = event.x()
            self._mouseY = event.y()

    def mousePressEvent(self, event):
        self._mouseX = event.x()
        self._mouseY = event.y()


# Renderer class to create the UI of the renderer.
class Renderer(object):
    def setupUi(self, window):
        window.setObjectName("Micro-CT Scan Renderer")
        window.resize(200, 330)

        # Views
        self.label = QtWidgets.QLabel(window)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setGeometry(QtCore.QRect(10, 10, 180, 20))
        self.label.setText("View Translation Coordinates")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { background-color : #dbd9d9;}")

        # Add X Y Z inputs below label
        self.label = QtWidgets.QLabel(window)
        self.label.setGeometry(QtCore.QRect(10, 40, 20, 20))
        self.label.setText("X:")

        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label = QtWidgets.QLabel(window)
        self.label.setGeometry(QtCore.QRect(10, 70, 20, 20))
        self.label.setText("Y:")
        
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label = QtWidgets.QLabel(window)
        self.label.setGeometry(QtCore.QRect(15, 100, 20, 20))
        self.label.setText("Z:")


        # Angles
        self.label2 = QtWidgets.QLabel(window)
        self.label2.setFrameShape(QtWidgets.QFrame.Box)
        self.label2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label2.setGeometry(QtCore.QRect(10, 150, 180, 20))
        self.label2.setText("Angle and Axis of Rotation")
        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setStyleSheet("QLabel { background-color : #dbd9d9;}")
        
        
        # Add X Y Z inputs below label2

        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label2 = QtWidgets.QLabel(window)
        self.label2.setGeometry(QtCore.QRect(15, 180, 20, 20))
        self.label2.setText(chr(176)+":")
        self.addInputs(window)

        self.label2 = QtWidgets.QLabel(window)
        self.label2.setGeometry(QtCore.QRect(10, 210, 20, 20))
        self.label2.setText("X:")

        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label2 = QtWidgets.QLabel(window)
        self.label2.setGeometry(QtCore.QRect(10, 240, 20, 20))
        self.label2.setText("Y:")
        
        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label2 = QtWidgets.QLabel(window)
        self.label2.setGeometry(QtCore.QRect(15, 270, 20, 20))
        self.label2.setText("Z:")
        self.addInputs(window)

        # Add Generate View button
        self.pushButton = QtWidgets.QPushButton(window)
        self.pushButton.setGeometry(QtCore.QRect(15, 300, 170, 20))
        self.pushButton.setText("Generate View")
        # Make the button call the generateView function when clicked
        self.pushButton.clicked.connect(self.generateView) 
    
    # Add input lines for the coordinates and angles
    def addInputs(self, window):
        self.lineEdit = QtWidgets.QLineEdit(window)
        self.lineEdit.setGeometry(QtCore.QRect(40, 40, 150, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("0")

        self.lineEdit_2 = QtWidgets.QLineEdit(window)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, 70, 150, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setText("0")

        self.lineEdit_3 = QtWidgets.QLineEdit(window)
        self.lineEdit_3.setGeometry(QtCore.QRect(40, 100, 150, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setText("0")

        # Angle
        self.lineEdit_4 = QtWidgets.QLineEdit(window)
        self.lineEdit_4.setGeometry(QtCore.QRect(40, 180, 150, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_4.setText("0")

        # Angles
        self.lineEdit_5 = QtWidgets.QLineEdit(window)
        self.lineEdit_5.setGeometry(QtCore.QRect(40, 210, 150, 20))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setText("0")

        self.lineEdit_6 = QtWidgets.QLineEdit(window)
        self.lineEdit_6.setGeometry(QtCore.QRect(40, 240, 150, 20))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_6.setText("0")


        self.lineEdit_7 = QtWidgets.QLineEdit(window)
        self.lineEdit_7.setGeometry(QtCore.QRect(40, 270, 150, 20))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_7.setText("0")

    # Generate the view given coordinates and angles
    def generateView(self):
        from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt

        # Get all coords and angles
        self.xCoord = self.lineEdit.text()
        self.yCoord = self.lineEdit_2.text()
        self.zCoord = self.lineEdit_3.text()
        self.Angle = self.lineEdit_4.text()
        self.xAxis = self.lineEdit_5.text()
        self.yAxis = self.lineEdit_6.text()
        self.zAxis = self.lineEdit_7.text()

        # Check if all coords and angles are numbers:
        try:
            float(self.xCoord)
            float(self.yCoord)
            float(self.zCoord)
            float(self.Angle)
            float(self.xAxis)
            float(self.yAxis)
            float(self.zAxis)
        except ValueError:
            print("One or more of the inputs is not a number.")
            return

        # Using sys run python3 test.py
        import subprocess
        # subprocess.call(["python3", "test.py", self.xCoord, self.yCoord, self.zCoord, self.xAngle, self.yAngle, self.zAngle])
        subprocess.call(["python3", "instant-NGP-cunerf/model/main.py", "--rhino", "--fp16", "--workspace \"cunerf-rhiner\"", "--render_new_view", "--translation", self.xCoord, self.yCoord, self.zCoord, "--rotation_angle", self.Angle, "--rotation_axis", self.xAxis, self.yAxis, self.zAxis])

        self.subwindow = QMainWindow()
        self.subwindow.setWindowTitle("Generated View")
        self.subwindow.resize(800, 800)

        # Create a QGraphicsScene and add the image to it
        scene = QGraphicsScene()
        pixmap = QPixmap("rendered_image.png")
        scene.addPixmap(pixmap)

        # Create a GraphicsViewWithZoom to display the QGraphicsScene
        self.view = GraphicsViewWithZoom(scene)
        layout = QVBoxLayout()
        layout.addWidget(self.view)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.subwindow.setCentralWidget(central_widget)

        self.resetScale = QtWidgets.QPushButton(self.subwindow)
        self.resetScale.setGeometry(QtCore.QRect(15, 15, 100, 20))
        self.resetScale.setText("Reset Zoom")
        self.resetScale.clicked.connect(self.view.resetScale) 
        self.subwindow.show()
 
#  Create window and generate the UI.
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
 
    MainWindow = QtWidgets.QMainWindow()
    ui = Renderer()
    
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("Renderer")
    MainWindow.show()
    sys.exit(app.exec_())