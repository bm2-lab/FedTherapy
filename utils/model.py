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
    finishSave = 0
    print(f"\rTraining: 0/{taskCount}", end='')
    while any(x.is_alive() for x in threadPool) or not outputQueue.empty():
        finish = 0
        while not outputQueue.empty():
            res.append(outputQueue.get())
            finish += 1
        if finish != 0:
            finishSave += finish
            print(f"\rTraining: {finishSave}/{taskCount}", end='')
        time.sleep(0.1)
    print()
    return res
