import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
app = QtGui.QApplication([])

class SingleArrayWindow:
    def __init__(self, sz):
        self.array = np.zeros(sz)
        plt = pg.plot()
        plt.setWindowTitle('mfcc')
        #plt.autoPixelRange = True
        plt.setRange(QtCore.QRectF(0, 0, 20, 20)) 
        plt.setLabel('bottom', 'Channel', units='')
        c = plt.plot(pen='#00FF00')
        self.curve = c
        

    def set_data(self, arr):
        self.array[:] = arr[:]
        self.curve.setData(self.array)
        app.processEvents()
