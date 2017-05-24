import Balance_error as BE
import random
import time
import sys
import numpy as np
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


trainrow1 =[]
result = open('/home/kalyani/Desktop/Stack_result.csv', 'w')
k = 0
m = 1

svmst = []
nbst = []
rfst = []
dtst = []
xgst = []
gst = []
vst = []

def creatingDataset(path1,path2):

    # opening files
    num = []
    traindata1 = []
    traindata2 = []
    traindata = []
    labels1 = []
    labels2 = []
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
    #print(len(data),len(traininglabels))
    for i in range(len(data)):
        if i in traininglabels:
            traindata.append(data[i])
            labels.append(traininglabels.get(i))
            if i%2 == 0:
                traindata1.append(data[i])
                labels1.append(traininglabels.get(i))
            else:
                traindata2.append(data[i])
                labels2.append(traininglabels.get(i))
        else:
            testdata.append(data[i])
            num.append(i)
    #print(len(traindata1), len(traininglabels),len(testdata))
    #print(traininglabels)
    #print(labels1)
    #print(labels2)

    return traindata1,traindata2,labels1,labels2,testdata,traindata,labels,num

def readStackfile():
    # opening files
    traindata = []
    testdata = []
    file1 = sys.argv[3]
    datafile = open(file1)
    data_line = datafile.readline()
    # reading dataset files
    while (data_line != ''):
        data_value = data_line.split()
        # data_value.append('1')
        data_value_no = []
        for i in range(len(data_value)):
            data_value_no.append(float(data_value[i]))
        # creating data matrix
        traindata.append(data_value)
        data_line = datafile.readline()
    datafile.close()

    file2 = sys.argv[4]
    datafile2 = open(file2)
    data_line2 = datafile2.readline()
    # reading dataset files
    while (data_line2 != ''):
        data_value = data_line2.split()
        # data_value.append('1')
        data_value_no = []
        for i in range(len(data_value)):
            data_value_no.append(float(data_value[i]))
        # creating data matrix
        testdata.append(data_value)
        data_line2 = datafile2.readline()
    datafile2.close()


    return testdata, traindata

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


def predictLabels(traindata, trainlabels, traintestdata,testdata):
    print("Classifying data",sep='',end='',flush=True)
    metatraindata = []
    metatestdata =  []
    votetestdata = []
    voteoutput = []
    global k

    tollist = [0.000001,0.000001,0.000001,0.000000001,0.000001,0.000001,0.000000001,0.0000000001,0.000001,0.0000001,0.0000001,0.000000001,0.0000001,0.00000001,0.0000001,
               0.00000001,0.000000001,0.000001,   0.00000001,  0.000001,0.000001,0.000001,0.000001, 0.00001,0.000001,0.000001,0.000001,0.000001,0.000001,0.0000001,
               0.0001, 0.000001,0.0000001,0.000001,0.000001,0.00000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,
               0.00000001,0.000001,0.000001,0.000001, 0.000001,0.000001,0.000001,0.000001]
    weight = [None,None,None,None,'balanced','balanced',None,'balanced',None,None,None,'balanced','balanced','balanced','balanced','balanced',None,
              'balanced',None,'balanced','balanced','balanced','balanced',None,'balanced','balanced','balanced','balanced','balanced',
              'balanced','balanced','balanced','balanced',    'balanced','balanced',None,'balanced','balanced','balanced','balanced','balanced','balanced'
        , 'balanced','balanced','balanced','balanced','balanced','balanced','balanced','balanced','balanced',None]
    c = [0.1,0.01,0.01,0.01,0.01,0.01,0.1,0.01,0.001,0.1,0.1,0.01,0.01,0.01,0.1,0.1,1,1,1,1,1,1,1,0.01,1,1,1,1,1,1,0.1,1,0.1,1,1,0.01,1,1,1,1,1,1,1,1,0.1,1,1,1,1,1,1,1]
    rate = [0.1,0.1,0.1,0.1,0.1,0.1,1.0,0.01,0.1,0.01,0.001,0.001,0.001,0.01,0.1,0.1,0.1,0.1,  0.0001,   0.1,0.1,0.1,0.1,0.001,0.1,0.1,0.1,0.1,0.1,0.1,0.001,0.1,1.0,0.1,
            0.1,0.0001,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01,0.1,0.1,0.1,0.1,0.1,0.1,0.001]
    gamma = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,10,0.0,0.0,10,1.0,10,0.0,0.1,0.0,1,0.0,10.0,0.0,0.0,0.0,0.0,10.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,10.0,0.0,0.0,10.0,0.0,
             0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.0,0.0,0.0,0.0,0.0,0.0,0.0,10.0]

    print("\n")
    print(tollist[8],c[8],weight[8],rate[8],gamma[8])
    print("k : ",k)

    svc = LinearSVC(tol=tollist[k],C=c[k])
    svc_clf = CalibratedClassifierCV(svc).fit(traindata,trainlabels)
    dt_clf = DecisionTreeClassifier(class_weight= weight[k]).fit(traindata,trainlabels)
    nb_clf = GaussianNB().fit(traindata,trainlabels)
    rf_clf = RandomForestClassifier(n_estimators=100).fit(traindata,trainlabels)
    xg_clf = XGBClassifier(learning_rate =rate[k], n_estimators=150, max_depth=10,
 min_child_weight=1, gamma=gamma[k], subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27).fit(traindata,trainlabels)

    if len(traintestdata) != 0:
        trsvc_predict = svc_clf.predict((traintestdata))
        trdt_predict = dt_clf.predict((traintestdata))
        trnb_predict = nb_clf.predict((traintestdata))
        trrf_predict = rf_clf.predict((traintestdata))
        trxg_predict = xg_clf.predict((traintestdata))



    svc_predict = svc_clf.predict((testdata))
    dt_predict =  dt_clf.predict((testdata))
    nb_predict = nb_clf.predict((testdata))
    rf_predict = rf_clf.predict((testdata))
    xg_predict = xg_clf.predict((testdata))


    model = [svc_predict, dt_predict, nb_predict, rf_predict, xg_predict]

    if len(traintestdata)== 0 :
        for i in range(0, len(testdata), 1):
            count = Counter([mod[i] for mod in model])
            voteoutput.append(count.most_common(1)[0][0])
        if m % 10 == 0:
            k += 1
        return voteoutput

    print("/n")

    predictedlabels_file = open('/home/kalyani/PycharmProjects/machine learning/trprediction.txt', 'w')
    predictedlabels_file1 = open('/home/kalyani/PycharmProjects/machine learning/prediction.txt', 'w')

    trmodel = [trsvc_predict, trdt_predict, trnb_predict, trrf_predict, trxg_predict]

    for i in range(0, len(traintestdata), 1):
        metatraindata.append([ mod[i] for mod in trmodel])
        predictedlabels_file.write(str(trsvc_predict[i])+" "+str(trdt_predict[i])
                                    +" "+str(trnb_predict[i])+" "+str(trrf_predict[i])+'\n')



    for i in range(0, len(testdata), 1):
        metatestdata.append([mod[i] for mod in model])
        predictedlabels_file1.write(str(svc_predict[i])+" "+str(dt_predict[i])
                                    +" "+str(nb_predict[i])+" "+str(rf_predict[i])+'\n')
    #print(metatestdata)
    predictedlabels_file1.close()
    predictedlabels_file.close()

    return svc_predict,dt_predict,nb_predict,rf_predict,xg_predict,metatraindata,metatestdata

def predictoutput(traindata, trainlabels, testdata,num):
    log_clf = LogisticRegression(tol=0.00001).fit(traindata, trainlabels)
    log_predict = log_clf.predict((testdata))
    predictedlabels_file1 = open('/home/kalyani/PycharmProjects/machine learning/output.txt', 'w')
    for i in range(len(testdata)):
        predictedlabels_file1.write(str(log_predict[i])+" "+str(num[i])+'\n')
    predictedlabels_file1.close()
    return log_predict,True

def main(path1,path2,path3,name):
    print(path1,path2,path3)
    startTime = time.time()
    global m
    global svmst
    global nbst
    global rfst
    global dtst
    global xgst
    global gst
    global vst


    single_class_prediction = {}
    traindata1,traindata2,labels1,labels2,testdata,data,labels,num = creatingDataset(path1,path2);

    traindata1 = np.array(traindata1).astype(np.float)
    traindata2 = np.array(traindata2).astype(np.float)
    labels1 = np.array(labels1).astype(np.float)
    labels2 = np.array(labels2).astype(np.float)
    testdata = np.array(testdata).astype(np.float)
    data = np.array(data).astype(np.float)
    labels = np.array(labels).astype(np.float)

    svc_predict,dt_predict,nb_predict,rf_predict,xg_predict,metatraindata,metatestdata = predictLabels(traindata1, labels1, traindata2,testdata)
    models = {'SVM':svc_predict,'Decision_Tree':dt_predict,'Naive_Bayes':nb_predict,'Random_Forest':rf_predict,'xgboost':xg_predict}
    trueclass = readTrueclass(path3)
    print("\n",name,":")
    for key,val in models.items():
        for i in range(len(val)):
            single_class_prediction[num[i]] = val[i]
        print("\n")
        print(key,": ")
        error =BE.balance_error(single_class_prediction,trueclass)
        if key == 'SVM':
            svmst.append(error)
        if key == 'Decision_Tree':
            dtst.append(error)
        if key == 'Naive_Bayes':
            nbst.append(error)
        if key == 'Random_Forest':
            rfst.append(error)
        if key == 'xgboost':
            xgst.append(error)

    if m % 10 == 0:
        result.write(str(np.mean(svmst))+","+str(np.mean(dtst))+","+str(np.mean(nbst))+","+str(np.mean(rfst))+","+str(np.mean(xgst))+",")
        svmst = []
        nbst = []
        rfst = []
        dtst = []
        xgst = []




    metatraindata = np.array(metatraindata).astype(np.float)
    metatestdata = np.array(metatestdata).astype(np.float)
    log_predict,isPred = predictoutput(metatraindata, labels2, metatestdata,num)
    #print(log_predict)
    for i in range(len(log_predict)):
        single_class_prediction[num[i]] = log_predict[i]
    print("\n")
    print("Generalize output: ")
    error = BE.balance_error(single_class_prediction, trueclass)
    gst.append(error)
    if m % 10 == 0:
        print("GST : ",gst)
        result.write(str(np.mean(gst)) + ",")
        gst = []

    print("\n")
    print("vote output: ")
    a = []
    vote_predict  = predictLabels(data, labels, a,testdata)
    #print(vote_predict)
    for i in range(len(log_predict)):
        single_class_prediction[num[i]] = vote_predict[i]
    error = BE.balance_error(single_class_prediction, trueclass)
    vst.append(error)
    if m % 10 == 0:
        print("VST :",vst)
        result.write(str(np.mean(vst)) + ",")
        result.write("\n")
        vst = []

    if (isPred):
        if m % 10 == 0:
            m = 0
        m += 1
        print("#" * 80)
        print("####### Data classified successfully and stored in 'predictedlabels' file ######")
        print("#" * 80)

    totalTime = (time.time()-startTime)

    print("Total Execution Time: ",int(totalTime/60), "minutes",int(totalTime%60),"seconds")


#main("/home/kalyani/Desktop/datasets_v1/antivirus/data","/home/kalyani/Desktop/datasets_v1/antivirus/random_class.1","/home/kalyani/Desktop/datasets_v1/antivirus/trueclass")