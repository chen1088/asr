import pyaudio  
import wave  
import numpy as np
import struct
from singlearraywindow import SingleArrayWindow
from peak_detection import get_filterbanks, powspec

nwindow = 2048  
stride = 128
nfeat = 20

f = wave.open(r"i7-965-clipped.wav","rb")   
p = pyaudio.PyAudio()  
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)
print(f.getframerate())    
data = f.readframes(stride)
frame = np.zeros(nwindow,dtype=float) 
count = 100000
# prepare mfcc filterbanks
fbs = get_filterbanks(nfeat,nwindow,f.getframerate(),0,False)
fcwindow = SingleArrayWindow(nfeat)

while data:  
    frame[:-stride] = frame[stride:]
    # unpack
    data_float = struct.unpack("%ih" % stride, data)
    data_float = [float(val) / pow(2, 15) for val in data_float]
    frame[-stride:] = data_float[:]
    frames = [frame]
    # get power spec
    pspec = powspec(frames, nwindow)
    # get mfcc feature
    feat = np.dot(pspec,fbs.T)
    # sync ui
    fcwindow.set_data(feat)
    stream.write(data)  
    data = f.readframes(stride)
    count = count - 1
    if count == 0:
        break
 
stream.stop_stream()  
stream.close() 
p.terminate()  