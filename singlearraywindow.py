import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore

class SingleArrayWindow:
    def __init__(self):
        self.app = QtGui.QApplication([])
        plt = pg.plot()
        plt.setWindowTitle('mfcc')
        #plt.autoPixelRange = True
        #plt.setRange(QtCore.QRectF(0, -20, 20, 40)) 
        plt.setRange(QtCore.QRectF(0,0,100,300))
        plt.setLabel('bottom', 'Channel', units='')
        c = plt.plot(pen='#00FF00')
        self.curve = c
        self.buffer = np.zeros(100)
        
    def push_point(self, datapoint):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = datapoint
        self.curve.setData(self.buffer.flatten())
        self.app.processEvents()

    def set_data(self, arr):
        self.curve.setData(arr.flatten())
        self.app.processEvents()
