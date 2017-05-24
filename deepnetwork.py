# Create first network with Keras
import Balance_error as BE
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras import callbacks
from sklearn.preprocessing import normalize
from collections import Counter
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model

result = open('/home/kalyani/Desktop/mydeepnetwork.csv', 'w')
k= 0
m = 0

def creatingDataset(path1,path2):

    # opening files
    num = []
    traindata = []
    testdata = []
    labels = []
    #file1 = sys.argv[1]
    datafile = open(path1)
    data = []
    data_line = datafile.readline()
    # reading dataset files
    while (data_line != ''):
        data_value = data_line.split()
        #data_value.append('1')
        data_value_no = []
        for i in range(len(data_value)):
            data_value_no.append(float(data_value[i]))
        # creating data matrix
        data.append(data_value)
        data_line = datafile.readline()
    datafile.close()

    # reading traininglabels file
    #file2 = sys.argv[2]
    lablefile = open(path2)
    traininglabels = {}
    lable_values = []
    lable_line = lablefile.readline()
    while (lable_line != ''):
        lable_values = lable_line.split()
        traininglabels[int(lable_values[1])] = int(lable_values[0])
        if (traininglabels[int(lable_values[1])] == 0):
            traininglabels[int(lable_values[1])] = 0
        lable_line = lablefile.readline()
    lablefile.close()
    print(len(data),len(traininglabels))
    for i in range(len(data)):
        if i in traininglabels:
            traindata.append(data[i])
            labels.append(traininglabels.get(i))
        else:
            testdata.append(data[i])
            num.append(i)

    return traindata,labels,testdata,num

def readTrueclass(path3):
    #file1 = sys.argv[3]
    lablefile = open(path3)
    traininglabels = {}
    lable_line = lablefile.readline()
    while (lable_line != ''):
        lable_values = lable_line.split()
        traininglabels[int(lable_values[1])] = int(lable_values[0])
        lable_line = lablefile.readline()
    # print(traininglabels)
    lablefile.close()
    return traininglabels



def network(traindata,labels,testdata,name,testclass):
    print(k,name)

    global k
    global m
    activation = ['sigmoid','relu','softmax','tanh']
    act1 = ['relu','relu','relu','sigmoid','relu','relu','tanh','relu','relu','tanh','relu','relu','relu','relu','sigmoid','tanh','relu','sigmoid','relu','relu',
            'relu','relu','relu','sigmoid','relu','relu','tanh','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','tanh','relu','relu',
            'relu','relu','relu','relu','relu','relu','relu','tanh','tanh','relu','tanh','relu']
    act2 = ['relu','relu','relu','sigmoid','relu','relu','tanh','relu','relu','tanh','relu','relu','relu','relu','sigmoid','tanh','relu','sigmoid','relu','relu',
            'relu','relu','relu','sigmoid','relu','relu','tanh','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','tanh','relu','relu','relu',
            'relu','relu','relu','relu','relu','relu','tanh','tanh','relu','tanh','relu']
    act3 = ['','','','sigmoid','','','','','','','','','','','sigmoid','','','','','','','','','','','','','','','','relu','relu','relu','','','relu','','','','','','',
            '','','','','','','','','','','']
    act4 = ['sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid',
            'sigmoid','sigmoid','tanh','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid',
            'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid',
            'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid']
    epoch = [150,200,200,500,200,200,200,300,300,200,200,200,200,200,200,200,300,3000,200,200,200,200,200,200,200,300,300,200,200,300,200,300,300,200,200,2000,200,
             200,500,200,100,300,300,300,1000,200,200,400,1000,200,300,200,200]
    print("k ",k)
    dataset = ['breast_cancer','ozone','parkinson2']
    model = Sequential()
    model.add(Dense(12, input_dim=len(traindata[0]), kernel_initializer='uniform', activation=act1[k]))
    model.add(Dense(8, kernel_initializer='uniform', activation=act2[k]))
    if name in dataset:
        print("i m here")
        model.add(Dense(4, kernel_initializer='uniform', activation=act3[k]))
    model.add(Dense(1, kernel_initializer='uniform', activation=act4[k]))

    # Compile model
    sgd = SGD(lr=0.1,decay=1e-7)
    opt = [sgd,sgd,sgd,'adamax','adamax','adam','adamax','adamax','adamax','adamax','adamax','adam','adamax','adam','adam','adam','adamax','adamax','adamax',sgd
        ,sgd,sgd,sgd,sgd,'adamax','adamax','adam','adamax','adamax','adamax','adamax','adamax','adam','adam','adamax','adam',sgd,'adamax','adam','adam','adam',
           'adam','adamax','adamax','adamax','adamax','adamax','adam','adam','adamax','adam','adam','adamax']
    model.compile(loss='binary_crossentropy', optimizer=opt[k], metrics=['accuracy'])
    # Fit the model
    er_st=callbacks.EarlyStopping()
    reduce_lr = callbacks.ReduceLROnPlateau(min_lr=0.01,patience=5)
    dataset1 = ['lsvt','steel_faults','student_alcohol','planning_relax','eeg_eye_state','susy']
    if name in dataset1:
        history = model.fit(traindata, labels, epochs=epoch[k], batch_size=10,validation_split=0.1,verbose=1,callbacks=[er_st])
    else:
        history = model.fit(traindata, labels, epochs=epoch[k], batch_size=10, validation_split=0.1, verbose=1)

    # calculate prediction0
    predictions = model.predict(testdata)
    m += 1

    if m%10 == 0:
        k += 1

    score = model.evaluate(testdata, testclass, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]*100)
    print(len(act1),len(act2),len(act3),len(act4),len(epoch),len(opt))
    # round predictions
    rounded  = [round(x[0]) for x in predictions]

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    #for i in range(0, len(testdata), 1):
        #print(rounded[i],num[i])
    return rounded

def main(path1,path2,path3,name):
    startTime = time.time()
    testclass = []

    global m
    single_class_prediction = {}
    data, labels,testdata,num = creatingDataset(path1, path2);
    print(len(data[0]))
    testdata = np.array(testdata).astype(np.float)
    #testdata = normalize(testdata, norm='l2', axis=0)
    data = np.array(data).astype(np.float)
    #data = normalize(data, norm='l2', axis=0)
    labels = np.array(labels).astype(np.float)
    trueclass = readTrueclass(path3)

    for i in num:
        testclass.append(trueclass.get(i))

    print(len(testclass),len(testdata))
    predication = network(data, labels, testdata,name,testclass)

    for i in range(len(predication)):
        single_class_prediction[num[i]] = predication[i]

    error = BE.balance_error(single_class_prediction, trueclass)

    result.write(str(error) + ",")
    if m % 10 == 0:
        m = 0
        result.write("\n")




def release_list(a):
   del a[:]
   del a