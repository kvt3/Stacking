
import sys

def balance_error(outtraininglabels,traininglabels):
    #####################################
    # CREATING DICTIONARY FOR LABELFILE #
    #####################################
    #file1 = sys.argv[1]

    '''file1 = path1
    lablefile = open(file1)
    traininglabels = {}
    lable_line = lablefile.readline()
    while (lable_line != ''):
        lable_values = lable_line.split()
        traininglabels[int(lable_values[1])] = int(lable_values[0])
        lable_line = lablefile.readline()
    #print(traininglabels)
    lablefile.close()

    ######################################
    # CREATING DICTIONARY FOR OUTPUTFILE #
    ######################################
    #file2 = sys.argv[2]
    file2 = path2
    outlablefile = open(file2)
    outtraininglabels = {}
    lable_line_out = outlablefile.readline()
    while (lable_line_out != ''):
        lable_line_out = lable_line_out.replace("[",'')
        lable_line_out = lable_line_out.replace("]",'')
        out_lable_values = lable_line_out.split()
        outtraininglabels[int(out_lable_values[1])] = int(out_lable_values[0])
        lable_line_out = outlablefile.readline()
    #print(outtraininglabels)
    outlablefile.close()'''


    ###############################################
    # CALCULATING TRUE AND FALSE RESULT OF OUTPUT #
    ###############################################
    class0_true=0
    class0_fasle=0
    class1_true=0
    class1_false=0
    for key in outtraininglabels:
        if(traininglabels.get(key) == 0):
            if(outtraininglabels.get(key) == 0):
                class0_true += 1
            else:
                class0_fasle +=1

        else:
            if(outtraininglabels.get(key)==1 and traininglabels.get(key) == 1):
                class1_true += 1
            else:
                class1_false += 1

    ###############################
    # CALCULATING BALANCE_ERROR #
    ##############################
    print("class0_false", class0_fasle)
    print("class0_true:", class0_true)
    print("class1_false:", class1_false)
    print("class1_true:", class1_true)
    if (int(class0_fasle) + int(class0_true)) != 0:
        balance_error1 = float(class0_fasle)/(int(class0_fasle) + int(class0_true))
    else:
        balance_error1 = 0
    if (int(class1_false) + int(class1_true)) != 0:
        balance_error2 = float(class1_false)/(int(class1_false) + int(class1_true))
    else:
        balance_error2 = 0

    balance_error = (balance_error1 + balance_error2)/2

    accuracy = ((class0_true + class1_true)/len(outtraininglabels)) * 100
    error = 100 - accuracy
    print("balance_error:",balance_error)
    print("accuracy : ", error)

    return error

#balance_error("/home/kalyani/Desktop/MS-Fall-2016/dataset/test/SNP/trainlabels", "/home/kalyani/PycharmProjects/machine learning/prediction")
