
import sys

import styleTransfer as st
from PIL import Image
import time

import threading

from PyQt5.QtWidgets import (QApplication, QDialog, QWidget,
        QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QTabWidget,
        QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QTextEdit,
        QVBoxLayout, QFileDialog)

from PyQt5.QtGui import QTextCursor, QIntValidator, QDoubleValidator

from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Dialog(QDialog):

    def __init__(self):
        super(Dialog, self).__init__()

        self.createMenu()
        self.createInputGroupBox()
        self.createParameterGroupBox()
        self.createLogGroupBox()
        self.createTabGroupBox()

        mainLayout = QHBoxLayout()        
        mainLayout.setMenuBar(self.menuBar)
        LeftLayout = QVBoxLayout()

        LeftLayout.addWidget(self.processGroupBox)
        LeftLayout.addWidget(self.formGroupBox)        
        LeftLayout.addWidget(self.logGroupBox)
        
        LeftLayout.setStretch(1,5)
        
        mainLayout.addLayout(LeftLayout)        
        mainLayout.addWidget(self.tabs)
        self.setLayout(mainLayout)

        self.setWindowTitle("Style Transfer")


    def createMenu(self):
        self.menuBar = QMenuBar()

        self.fileMenu = QMenu("&File", self)
        self.exitAction = self.fileMenu.addAction("&Exit")
        self.menuBar.addMenu(self.fileMenu)

        self.exitAction.triggered.connect(self.close)
        

    def createInputGroupBox(self):
        self.processGroupBox = QGroupBox("Input")
        layout = QGridLayout()
        
        self.contentButton = QPushButton("Select Content Image")
        self.contentPath = QLineEdit()
        self.contentPath.setFixedWidth(240)
        self.contentPath.setReadOnly(True)
        self.styleButton = QPushButton("Select Style Image")
        self.stylePath = QLineEdit()
        self.stylePath.setReadOnly(True)
        
        self.contentButton.clicked.connect(self.openContentImg)
        self.styleButton.clicked.connect(self.openStyleImg)
        
        layout.addWidget(self.contentButton, 0, 0)
        layout.addWidget(self.contentPath, 0, 1)
        layout.addWidget(self.styleButton, 1, 0)
        layout.addWidget(self.stylePath, 1, 1)

        #layout.setColumnStretch(1, 10)
        #layout.setColumnStretch(2, 20)
        self.processGroupBox.setLayout(layout)
        

    def createParameterGroupBox(self):
        
        self.formGroupBox = QGroupBox("Parameters")
        layout = QFormLayout()
        layout.setHorizontalSpacing(50)
        self.imgSizeEdit = QLineEdit('512')
        self.imgSizeEdit.setAlignment(Qt.AlignCenter)
        self.imgSizeEdit.setValidator(QIntValidator())
        layout.addRow(QLabel("Image Max Size:"), self.imgSizeEdit)
        self.itTimesEdit = QLineEdit('128')
        self.itTimesEdit.setAlignment(Qt.AlignCenter)
        self.itTimesEdit.setValidator(QIntValidator())
        layout.addRow(QLabel("Iteration Times:"), self.itTimesEdit)
        self.cntWeightEdit = QLineEdit('1000')
        self.cntWeightEdit.setAlignment(Qt.AlignCenter)
        self.cntWeightEdit.setValidator(QDoubleValidator())
        layout.addRow(QLabel("Content Weight:"), self.cntWeightEdit)
        self.stlWeightEdit = QLineEdit('0.01')
        self.stlWeightEdit.setAlignment(Qt.AlignCenter)
        self.stlWeightEdit.setValidator(QDoubleValidator())
        layout.addRow(QLabel("Style Weight:"), self.stlWeightEdit)
        
        self.transferButton = QPushButton("Start Transfer")
        layout.addRow(self.transferButton)
        
        self.transferButton.clicked.connect(self.startTransfer)
        self.formGroupBox.setLayout(layout)


    def createLogGroupBox(self):
        self.logGroupBox = QGroupBox("Data Log")
        layout = QHBoxLayout()

        self.logEdit = QTextEdit()
        self.logEdit.setFixedHeight(500)
        self.log('This is a style transfer demo with PyQt based UI.')

        layout.addWidget(self.logEdit)

        self.logGroupBox.setLayout(layout)
        
        
    def createTabGroupBox(self):
        self.tabs = QTabWidget() 
        self.tabs.setFixedWidth(1000)
        
        self.fig1, self.canvas1 = self.addPlotTab('Input and Output')
        
    def addPlotTab(self, name):
        
        tab = QWidget()
        tab.setFixedWidth(1000)
        self.tabs.addTab(tab,name)        
        #self.tabs.setFixedWidth(600)  
        
        layout = QVBoxLayout()
        
        fig = Figure()                
        canvas = FigureCanvas(fig)
        
        layout.addWidget(canvas)
        tab.setLayout(layout)
        
        return fig, canvas
        
    def log(self, logText):
        self.logEdit.append(logText+'\n')
        self.logEdit.moveCursor(QTextCursor.End)


    def openContentImg(self):        
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                "Open Content Image", "",
                "Image Files (*.jpg *.png *.bmp)", options=options)
        if fileName:
            self.contentImg = st.load_img(fileName).astype('uint8')
            self.contentPath.setText(fileName)
            self.log('Content Image:    '+fileName)
            self.axes1 = self.fig1.add_subplot(221, xticks=[], yticks=[],xlabel='Content Image')
            self.axes1.imshow(self.contentImg[0])
            self.canvas1.draw()
            
    def openStyleImg(self):        
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                "Open Style Image", "",
                "Image Files (*.jpg *.png *.bmp)", options=options)
        if fileName:
            self.styleImg = st.load_img(fileName).astype('uint8')
            self.stylePath.setText(fileName)
            self.log('Style Image:    '+fileName)
            self.axes2 = self.fig1.add_subplot(223, xticks=[], yticks=[],xlabel='Style Image')
            self.axes2.imshow(self.styleImg[0])
            self.canvas1.draw()
    
    def startTransfer(self):
        # Content layer where will pull our feature maps
        st.content_layers = ['block5_conv2'] 
        
        # Style layer we are interested in
        st.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1'
                       ]
        
        st.num_content_layers = len(st.content_layers)
        st.num_style_layers = len(st.style_layers)
        
        if self.tabs.count()<2:
            self.fig2, self.canvas2 = self.addPlotTab("Iteration Images")
            
        self.launchTransferThread()
        
        
    def launchTransferThread(self):
        t = threading.Thread(target=self.runStyleTransfer)
        t.start()
        
    def loadParameters(self):
        self.imgSize = int(self.imgSizeEdit.text())
        st.mSize = self.imgSize
        self.itTimes = int(self.itTimesEdit.text())
        self.contentWeight = float(self.cntWeightEdit.text())
        self.styleWeight = float(self.stlWeightEdit.text())

    def runStyleTransfer(self): 
        
        self.log('------------Style Transfer Start------------')
        content_path = self.contentPath.text()
        style_path = self.stylePath.text()
        
        self.loadParameters()
        
        model = st.get_model() 
        for layer in model.layers:
            layer.trainable = False
          
        # Get the style and content feature representations (from our specified intermediate layers) 
        style_features, content_features = st.get_feature_representations(model, content_path, style_path)
        gram_style_features = [st.gram_matrix(style_feature) for style_feature in style_features]
          
        # Set initial image
        init_image = st.load_and_process_img(content_path)
        init_image = st.tfe.Variable(init_image, dtype=st.tf.float32)
        # Create our optimizer
        opt = st.tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
        
        # Store our best result
        self.best_loss, self.best_img = float('inf'), None
          
        # Create a nice config 
        loss_weights = (self.styleWeight, self.contentWeight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }
        
        # For displaying
        num_rows = 4
        num_cols = 4
        display_interval = self.itTimes/(num_rows*num_cols)
        start_time = time.time()
        global_start = time.time()
          
        norm_means = st.np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means   
          
        self.tabs.setCurrentIndex(1)
          
        imgs = []
        for i in range(self.itTimes):
            grads, all_loss = st.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = st.tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            
            #print('display_interval:{}, i:{}'.format(display_interval,i))
        
            if loss < self.best_loss:
                # Update best loss and best image from total loss. 
                 self.best_loss = loss
                 self.best_img = st.deprocess_img(init_image.numpy())
        
            if i % display_interval== 0:
                start_time = time.time()
          
                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = st.deprocess_img(plot_img)
                
                label = "Iteration: "+str(i)
                axes = self.fig2.add_subplot(
                        num_rows, num_cols, int(i/display_interval)+1, 
                        xticks=[], yticks=[],xlabel=label)
                axes.imshow(plot_img)
                self.canvas2.draw()
                       
                self.log('Iteration: {}, Total loss: {:.4e}, ' 
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(i, loss, style_score, content_score, time.time() - start_time))
            else:
                self.log('Iteration: {}'.format(i))   
        
          
        print('Total time: {:.4f}s'.format(time.time() - global_start))
        st.IPython.display.clear_output(wait=True)
        plt.figure(figsize=(14,4))
        for i,img in enumerate(imgs):
            plt.subplot(num_rows,num_cols,i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
          
        Image.fromarray(self.best_img)
        
        self.tabs.setCurrentIndex(0)
        self.axes3 = self.fig1.add_subplot(122, xticks=[], yticks=[],xlabel='Result Image')
        self.axes3.imshow(self.best_img)
        self.canvas1.draw()
        
        self.log('------------Style Transfer Complete------------')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = Dialog()
    dialog.exec_()