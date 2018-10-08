
import wx 
import wx.lib.agw.aui as aui

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import tensorflow as tf
from tensorflow import keras

import numpy as np


class Plot(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))
        self.canvas = FigureCanvas(self, -1, self.figure)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)


class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self, name="plot"):
        page = Plot(self.nb)
        self.nb.AddPage(page, name)
        return page.figure
    
 
class wxDemo(wx.Frame):
    dispNull = 0
    dispTraining = 1
    dispTest = 2
    dispResult = 3
    def __init__(self, parent, title): 
       super(wxDemo, self).__init__(parent, title = title, pos=(200,200), size=(1000,600))
		
       panel = wx.Panel(self)        
       
       self.makeMenuBar()       
       
       self.CreateStatusBar()
       self.SetStatusText("Welcome to wxPython!")
       
       vbox = wx.BoxSizer(wx.HORIZONTAL) 		

       bboxSizer = self.makeButtonBox(panel)
       tboxSizer = self.makeTextBox(panel)
       dboxSizer = self.makeDisplayBox(panel)
       vbox.Add(bboxSizer,0, wx.ALL|wx.CENTER, 5) 
       vbox.Add(tboxSizer,0, wx.ALL|wx.CENTER, 5) 
       vbox.Add(dboxSizer,1, wx.EXPAND) 
       
       self.dispMode = wxDemo.dispNull
       
       panel.SetSizer(vbox) 
       self.Centre() 
                
       panel.Fit() 
       self.Show()  
       
    def makeTextBox(self, panel):
        txt = wx.StaticBox(panel, -1, 'Neural Network Parameters:') 
        txtSizer = wx.StaticBoxSizer(txt, wx.VERTICAL) 
       
        txtbox = wx.BoxSizer(wx.VERTICAL) 
       
        hlText = wx.StaticText(panel, -1, "Number of Hidden Layers:")		
        self.nHiddenLayers = wx.TextCtrl(panel, -1, style = wx.ALIGN_LEFT, value='1')
        uText = wx.StaticText(panel, -1, "Number of Units per Hidden Layer") 
        self.nUnits = wx.TextCtrl(panel, -1, style = wx.ALIGN_LEFT, value='128')         
        eText = wx.StaticText(panel, -1, "Number of Epochs") 
        self.nEpochs = wx.TextCtrl(panel, -1, style = wx.ALIGN_LEFT, value='5') 
         
        txtbox.Add(hlText, 0, wx.ALL|wx.CENTER, 5) 
        txtbox.Add(self.nHiddenLayers, 0, wx.ALL|wx.CENTER, 5)
        txtbox.Add(uText, 0, wx.ALL|wx.CENTER, 5) 
        txtbox.Add(self.nUnits, 0, wx.ALL|wx.CENTER, 5) 
        txtbox.Add(eText, 0, wx.ALL|wx.CENTER, 5) 
        txtbox.Add(self.nEpochs, 0, wx.ALL|wx.CENTER, 5) 
       
        txtSizer.Add(txtbox, 0, wx.ALL|wx.CENTER, 10)  
        return txtSizer
    
    def makeButtonBox(self, panel):
        sbox = wx.StaticBox(panel, -1, 'Process:') 
        sboxSizer = wx.StaticBoxSizer(sbox, wx.VERTICAL) 
		
        hbox = wx.BoxSizer(wx.VERTICAL) 
        ButtonLoad = wx.Button(panel, -1, 'Load Data') 
        ButtonDispTrain = wx.Button(panel, -1, 'Display Training Data') 
        ButtonDispTest = wx.Button(panel, -1, 'Display Test Data') 
        ButtonTrain = wx.Button(panel, -1, 'Start Training') 
        ButtonResult = wx.Button(panel, -1, 'Display Test Result') 
        #ButtonClear = wx.Button(panel, -1, 'Clear Model') 
		
        hbox.Add(ButtonLoad, 0, wx.ALL|wx.CENTER, 10) 
        hbox.Add(ButtonDispTrain, 0, wx.ALL|wx.CENTER, 10) 
        hbox.Add(ButtonDispTest, 0, wx.ALL|wx.CENTER, 10) 
        hbox.Add(ButtonTrain, 0, wx.ALL|wx.CENTER, 10) 
        hbox.Add(ButtonResult, 0, wx.ALL|wx.CENTER, 10) 
        #hbox.Add(ButtonClear, 0, wx.ALL|wx.CENTER, 10) 
        
        sboxSizer.Add(hbox, 0, wx.ALL|wx.LEFT, 10) 
        
        self.Bind(wx.EVT_BUTTON, self.OnLoadData, ButtonLoad)
        self.Bind(wx.EVT_BUTTON, self.OnDispTrainingData, ButtonDispTrain)
        self.Bind(wx.EVT_BUTTON, self.OnDispTestData, ButtonDispTest)
        self.Bind(wx.EVT_BUTTON, self.OnStartTraining, ButtonTrain)
        self.Bind(wx.EVT_BUTTON, self.OnDisplayResult, ButtonResult)
        return sboxSizer
    
    def makeDisplayBox(self, panel):        
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        self.plotter1 = PlotNotebook(panel)
        vbox.Add(self.plotter1, 1, wx.EXPAND)
        
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.ButtonLast = wx.Button(panel, -1, 'Last')
        hbox.Add(self.ButtonLast, 0, wx.ALL|wx.CENTER, 10)
        
        self.pText = wx.StaticText(panel, -1, "Page 1")	
        hbox.Add(self.pText, 0, wx.ALL|wx.CENTER, 10)
        
        self.ButtonNext = wx.Button(panel, -1, 'Next') 
        hbox.Add(self.ButtonNext, 0, wx.ALL|wx.CENTER, 10)        
        
        self.Bind(wx.EVT_BUTTON, self.OnLastPage, self.ButtonLast)
        self.Bind(wx.EVT_BUTTON, self.OnNextPage, self.ButtonNext)
        
        self.ButtonLast.Disable()
        self.ButtonNext.Disable()
        
        
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnShowPopup, self.plotter1)
        
        vbox.Add(hbox, 0, wx.ALL|wx.CENTER, 10)
        return vbox
    
    def makeMenuBar(self):
        fileMenu = wx.Menu()
        # The "\t..." syntax defines an accelerator key that also triggers
        # the same event
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()
        # When using a stock ID we don't need to specify the menu item's
        # label
        exitItem = fileMenu.Append(wx.ID_EXIT)

        # Now a help menu for the about item
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
        
        
    def plotTrainingData(self, page):
        if page>1:
            self.ButtonLast.Enable()
        else:
            self.ButtonLast.Disable()
            
        if 20*page<len(self.train_images):
            self.ButtonNext.Enable()
        else:
            self.ButtonNext.Disable()
            
        self.currentPage = page
        sz = 'Page ' + str(page)
        self.pText.SetLabel(sz)
        pc = self.plotter1.nb.GetPageCount()
        for i in range(pc):
            self.plotter1.nb.DeletePage(0)
        figure = self.plotter1.add('training data')
        figure.set_tight_layout(True)
        for i in range(20):
            ax1=figure.add_subplot(4,5,i+1,xlabel=self.class_names[self.train_labels[20*(page-1)+i]], xticks=[], yticks=[])
            ax1.imshow(self.train_images[20*(page-1)+i])
        
        
        
        
    def plotTestData(self, page):
        if page>1:
            self.ButtonLast.Enable()
        else:
            self.ButtonLast.Disable()
            
        if 20*page<len(self.test_images):
            self.ButtonNext.Enable()
        else:
            self.ButtonNext.Disable()
            
        self.currentPage = page
        sz = 'Page ' + str(page)
        self.pText.SetLabel(sz)
        pc = self.plotter1.nb.GetPageCount()
        for i in range(pc):
            self.plotter1.nb.DeletePage(0)
        figure = self.plotter1.add('test data')
        figure.set_tight_layout(True)
        for i in range(20):
            ax1=figure.add_subplot(4,5,i+1,xlabel=self.class_names[self.test_labels[20*(page-1)+i]], xticks=[], yticks=[])
            ax1.imshow(self.test_images[20*(page-1)+i])   
            
    def plotResultImg(self, page):        
        if page>1:
            self.ButtonLast.Enable()
        else:
            self.ButtonLast.Disable()
            
        if 20*page<len(self.test_images):
            self.ButtonNext.Enable()
        else:
            self.ButtonNext.Disable()
            
        self.currentPage = page
        sz = 'Page ' + str(page)
        self.pText.SetLabel(sz)
        
        pc = self.plotter1.nb.GetPageCount()
        for i in range(pc):
            self.plotter1.nb.DeletePage(0)
        figure = self.plotter1.add('test result')
        figure.set_tight_layout(True)
        for i in range(20):
            
            predictions_array, true_label, img = self.predictions[20*(page-1)+i], self.test_labels[20*(page-1)+i], self.test_images[20*(page-1)+i]
            
            predicted_label = np.argmax(predictions_array)
            if predicted_label == true_label :
                lcolor = 'blue'
            else:
                lcolor = 'red'
            
            sz = "{} {:2.0f}%\n({})".format(self.class_names[predicted_label],
                   100*np.max(predictions_array),
                   self.class_names[true_label])

            ax1=figure.add_subplot(4,5,i+1, xticks=[],yticks=[])
            ax1.imshow(img)
            ax1.set_xlabel(sz, color=lcolor)              


    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


    def OnHello(self, event):
        """Say hello to the user."""
        wx.MessageBox("Hello again from wxPython")


    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython sample for classification",
                      "About Classification Demo",
                      wx.OK|wx.ICON_INFORMATION)
        
    def OnShowPopup(self, event):
        print('Right Click')
        
    def OnLoadData(self, event):   
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        
        self.dispMode = wxDemo.dispTraining
        self.plotTrainingData(1)
        
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnShowPopup, self.plotter1.nb)
        
        sz = 'Loading succeeded! Number of training samples: ' + str(len(self.train_images))
        self.SetStatusText(sz)
        
        
    def OnDispTrainingData(self, event):                
        if hasattr(self, 'train_images'):
            self.dispMode = wxDemo.dispTraining
            self.plotTrainingData(1)
        
            sz = 'Number of training samples: ' + str(len(self.train_images))
            self.SetStatusText(sz)
        else:
            wx.MessageBox("Please Load Data First!", "No Data Loaded.", wx.OK)        
    
    
    def OnDispTestData(self, event):          
        if hasattr(self, 'test_images'):
            self.dispMode = wxDemo.dispTest
            self.plotTestData(1)
        
            sz = 'Number of test samples: ' + str(len(self.test_images))
            self.SetStatusText(sz)
        else:
            wx.MessageBox("Please Load Data First!", "No Data Loaded.", wx.OK)
    
    
    def OnStartTraining(self, event):        
        
        if hasattr(self, 'test_images'):
            
            self.ButtonNext.Disable()
            self.ButtonLast.Disable()
        
            sz = 'Compiling model......'
            self.SetStatusText(sz)
            
            try:
                HidLayerNum = int(self.nHiddenLayers.GetValue())
            except:
                HidLayerNum = 1            
            if HidLayerNum<0:
                HidLayerNum=0
            self.nHiddenLayers.SetValue(str(HidLayerNum))
            
            try:
                unitsNum = int(self.nUnits.GetValue())
            except:
                unitsNum = 128
            if unitsNum<1:
                unitsNum=1
            self.nUnits.SetValue(str(unitsNum))
            
            try:
                epochsNum = int(self.nEpochs.GetValue())
            except:
                epochsNum = 5
            if epochsNum<1:
                epochsNum=1
            self.nEpochs.SetValue(str(epochsNum))
            
            network = [
            keras.layers.Flatten(input_shape=(28,28)), 
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ]
            
            for i in range(HidLayerNum):
                network.insert(1,keras.layers.Dense(unitsNum, activation=tf.nn.relu))
            
                
            self.model = keras.Sequential(network)
    
            self.model.compile(optimizer=tf.train.AdamOptimizer(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            sz = 'Fitting model......'
            self.SetStatusText(sz)
            
            history = self.model.fit(self.train_images, self.train_labels, epochs=epochsNum)
            
            pc = self.plotter1.nb.GetPageCount()
            for i in range(pc):
                self.plotter1.nb.DeletePage(0)
            axes1 = self.plotter1.add('loss').gca(xticks = range(1, epochsNum+1), xlabel='Epochs', ylabel='loss')
            x = range(1, epochsNum+1)
            y = history.history['loss']
            axes1.plot(x, y)
            for i,j in zip(x,y):
                axes1.annotate("{:.4f}".format(j),xy=(i,j), xytext=(5,5), textcoords='offset points')
                
            x = range(1, epochsNum+1)
            y = history.history['acc']
            axes2 = self.plotter1.add('acc').gca(xticks = range(1, epochsNum+1), xlabel='Epochs', ylabel='accuracy')
            axes2.plot(x, y)
            for i,j in zip(x,y):
                axes2.annotate("{:.4f}".format(j),xy=(i,j), xytext=(5,5), textcoords='offset points')
            
            
            
            test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
            sz = 'Test accuracy: '+ str(test_acc)
            print(sz)
            self.SetStatusText(sz)
            
        else:
            wx.MessageBox("Please Load Data First!", "No Data Loaded.", wx.OK)
        
    def OnDisplayResult(self, event):
        
        if hasattr(self, 'model'):
            self.predictions = self.model.predict(self.test_images)
        
            self.dispMode = wxDemo.dispResult
            self.plotResultImg(1)
            
            sz = 'Number of test samples: ' + str(len(self.test_images))
            self.SetStatusText(sz)
        else:
            wx.MessageBox("Please Train Data First!", "No Data Trained.", wx.OK) 
        
    
    def OnLastPage(self, event):
        self.currentPage = self.currentPage-1
        if self.dispMode == wxDemo.dispTraining:
            self.plotTrainingData(self.currentPage)
        elif self.dispMode == wxDemo.dispTest:
            self.plotTestData(self.currentPage)
        elif self.dispMode == wxDemo.dispResult:
            self.plotResultImg(self.currentPage)
        
    def OnNextPage(self, event):
        self.currentPage = self.currentPage+1
        if self.dispMode == wxDemo.dispTraining:
            self.plotTrainingData(self.currentPage)
        elif self.dispMode == wxDemo.dispTest:
            self.plotTestData(self.currentPage)
        elif self.dispMode == wxDemo.dispResult:
            self.plotResultImg(self.currentPage)
        

app = wx.App() 
wxDemo(None,  'Classification Demo') 
app.MainLoop()