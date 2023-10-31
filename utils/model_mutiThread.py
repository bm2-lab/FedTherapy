import time
import pickle
import threading
import itertools
from queue import Queue

def listMult(li: list) -> int:
    i = 1
    for l in li:
        i *= len(l)
    return i

def pklSave(data, fileName):
    with open(fileName, "bw") as file:
        pickle.dump(data, file)
        
def pklLoad(fileName):
    with open(fileName, "br") as file:
        return pickle.load(file)

class MoniterThread(threading.Thread):
    def __init__(self, inputQueue, outputQueue, initFunc, trainingPipeline, usingThread=1,  **kargs):
        super(MoniterThread, self).__init__()
        initFunc(kargs)
        self.usingThread = usingThread
        self.iptQ = inputQueue
        self.optQ = outputQueue
        self.pipeline = trainingPipeline
        self.kargs = kargs
        
    def run(self):
        for i in range(self.usingThread):
            TrainingThread(self.inputQ, self.optQ, self.pipeline, **self.kargs).start()    

class TrainingThread(threading.Thread):
    def __init__(self, inputQueue, outputQueue, trainingPipeline, **kargs):
        super(TrainingThread, self).__init__()
        self.iptQ = inputQueue
        self.optQ = outputQueue
        self.pipeline = trainingPipeline
        self.kargs = kargs
        
    def run(self):
        while not self.iptQ.empty():
            currentParamater = self.iptQ.get()
            self.optQ.put((currentParamater, 
                           self.pipeline(**self.kargs, **currentParamater)))

def createThread(inputQueue, outputQueue, initFunc, trainingPipeline, usingThread=1,  **kargs):
    initFunc(kargs)
    for i in range(usingThread):
        yield TrainingThread(inputQueue, outputQueue, trainingPipeline, **kargs)

def gridSearch(stablePara, adjustPara, deviceLi, usingThreadNum, threadInitFunc, trainingPipeline):
    inputQueue = Queue()
    outputQueue = Queue()
    for item in itertools.product(*adjustPara.values()):
        inputQueue.put({k:v for k,v in zip(adjustPara, item)})
    taskCount = listMult(adjustPara.values())
        
    threadPool = []
    for d in deviceLi:
        for t in createThread(inputQueue, outputQueue, threadInitFunc, 
                              trainingPipeline, usingThread=usingThreadNum, 
                              device=d, **stablePara):
#        t = TrainingThread(inputQueue, outputQueue, threadInitFunc, trainingPipeline, device=d, **stablePara)
            threadPool.append(t)
            t.start()
        
    res = []
    while any(x.is_alive() for x in threadPool) or not outputQueue.empty():
        print(f"\rTraining: {taskCount-inputQueue.qsize()}/{taskCount}", end='')
        while not outputQueue.empty():
            res.append(outputQueue.get())
        time.sleep(1)
    print()
    return res
