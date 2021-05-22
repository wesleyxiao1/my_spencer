import pandas
import pickle
#from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

TRAIN_MIN = 3
TRAIN_MAX = 12000
VAL_MIN = 12003
VAL_MAX = 18000
TEST_MIN = 18003
TEST_MAX = 24000

def train_old(datapoints=30000, c=10, k='linear'):
    data = pandas.read_csv('data/train_model_data.csv')
    data = data.head(datapoints)

    X = data.drop(columns=['frameID','pedID1','pedID2','group_label'])
    y = data['group_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    logging.info("Starting to train SVM")
    logging.info(datapoints)
    svc = SVC(kernel=k, probability=True, class_weight='balanced', C=c)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)
    logging.info(confusion_matrix(y_test,y_pred))
    logging.info(classification_report(y_test,y_pred))

    logging.info("writing to outfile")
    outfile = "data/social_relations_" +str(datapoints)+ ".model"
    pickle.dump(svc, open(outfile, "wb"))
    logging.info("finished\n")

def train_svm(name, class_weight=None, c=1.0):
    data = pandas.read_csv('data/model_data.csv')
    #data = data.head(datapoints)

    train = data.loc[((data.frameID >= TRAIN_MIN) & (data.frameID <= TRAIN_MAX))]
    val = data.loc[((data.frameID >= VAL_MIN) & (data.frameID <= VAL_MAX))]
    test = data.loc[((data.frameID >= TEST_MIN) & (data.frameID <= TEST_MAX))]

    X_train = train.drop(columns=['frameID','pedID1','pedID2','group_label'])
    y_train = train['group_label']

    X_val = val.drop(columns=['frameID','pedID1','pedID2','group_label'])
    y_val = val['group_label']

    X_test = test.drop(columns=['frameID','pedID1','pedID2','group_label'])
    y_test = test['group_label']

    logging.info("Starting to train SVM")
    #clf = svm.LinearSVC(class_weight=class_weight, C=c)
    clf = SVC(kernel='linear', probability=True, class_weight=class_weight, C=c)
    clf.fit(X_train, y_train)

    logging.info("=======" + name + "=======")
    logging.info("Testing on val set")
    y_pred = clf.predict(X_val)
    logging.info(confusion_matrix(y_val,y_pred))
    logging.info(classification_report(y_val,y_pred))

    logging.info("Testing on test set")
    y_pred = clf.predict(X_test)
    logging.info(confusion_matrix(y_test,y_pred))
    logging.info(classification_report(y_test,y_pred))

    logging.info("writing to outfile")
    outfile = "data/social_relations_" +str(name)+ ".model"
    pickle.dump(clf, open(outfile, "wb"))
    logging.info("finished\n")

def train(name, class_weight=None, c=1.0, data='data/model_data.csv', training_only=True, drop=[]):
    data = pandas.read_csv(data)
    #data = data.head(datapoints)

    train = data.loc[((data.frameID >= TRAIN_MIN) & (data.frameID <= TRAIN_MAX))]
    val = data.loc[((data.frameID >= VAL_MIN) & (data.frameID <= VAL_MAX))]
    test = data.loc[((data.frameID >= TEST_MIN) & (data.frameID <= TEST_MAX))]

    if training_only:
        X_train = train.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_train = train['group_label']

        X_val = val.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_val = val['group_label']

        X_test = test.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_test = test['group_label']
    else:
        t = data.loc[((data.frameID >= TRAIN_MIN) & (data.frameID <= VAL_MAX))]
        X_train = t.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_train = t['group_label']

        X_val = val.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_val = val['group_label']

        X_test = test.drop(columns=['frameID','pedID1','pedID2','group_label'] + drop)
        y_test = test['group_label']
    #import pdb; pdb.set_trace()
    logging.info("Starting to train SVM")
    m = svm.LinearSVC(class_weight=class_weight, C=c)
    clf = CalibratedClassifierCV(m)
    clf.fit(X_train, y_train)

    logging.info("=======" + name + "=======")
    logging.info("Testing on val set")
    y_pred = clf.predict(X_val)
    logging.info(confusion_matrix(y_val,y_pred))
    logging.info(classification_report(y_val,y_pred))

    logging.info("Testing on test set")
    y_pred = clf.predict(X_test)
    logging.info(confusion_matrix(y_test,y_pred))
    logging.info(classification_report(y_test,y_pred))

    logging.info("writing to outfile")
    outfile = "data/social_relations_" +str(name)+ ".model"
    pickle.dump(clf, open(outfile, "wb"))
    logging.info("finished\n")

'''
train(c=100, class_weight='balanced', name="penalty_100_balanced")
train(c=10, class_weight='balanced', name="penalty_10_balanced")
train(c=1, class_weight='balanced', name="penalty_1_balanced")
train(c=0.1, class_weight='balanced', name="penalty_0.1_balanced")
train(c=0.01, class_weight='balanced', name="penalty_0.01_balanced")
train(c=0.001, class_weight='balanced', name="penalty_0.001_balanced")
train(c=0.0001, class_weight='balanced', name="penalty_0.0001_balanced")
'''

'''
train(c=100,    class_weight=None, name="penalty_100_norm", data="data/model_data_normalized.csv")
train(c=10,     class_weight=None, name="penalty_10_norm", data="data/model_data_normalized.csv")
train(c=1,      class_weight=None, name="penalty_1_norm", data="data/model_data_normalized.csv")
train(c=0.1,    class_weight=None, name="penalty_0.1_norm", data="data/model_data_normalized.csv")
train(c=0.01,   class_weight=None, name="penalty_0.01_norm", data="data/model_data_normalized.csv")
train(c=0.001,  class_weight=None, name="penalty_0.001_norm", data="data/model_data_normalized.csv")
'''

c_values = [100, 10, 1, 0.1, 0.01, 0.001]
datasets = ["data/model_data.csv", "data/model_data_normalized.csv"]
weights  = [None, "balanced"]
'''
# normalized data
for c in c_values:
    name = "pen_"+str(c)+"_norm"
    train(c=c, class_weight=None, name=name, data="data/model_data_normalized.csv")
'''
'''
# train in training and validation
for c in c_values:
    name = "pen_"+str(c)+"_norm_train+valid"
    train(c=c, class_weight=None, name=name, data="data/model_data_normalized.csv", training_only=False)
    name = "pen_"+str(c)+"_unnorm_train+valid"
    train(c=c, class_weight=None, name=name, data="data/model_data.csv", training_only=False)
'''
#train(c=1, class_weight='balanced', name="penalty_1_norm_angle", data="data/model_data_normalized.csv", drop=["dist", "speed"])
train(name="01_group_labels", data="data/model_data_01_group_labels.csv")
train(name="norm_per_frame", data="data/model_data_norm_per_frame.csv")
train(name="norm_per_frame_01_group_labels", data="data/model_data_norm_per_frame_01_group_labels.csv")

#train_svm(c=1, class_weight=None, name="penalty_1_SVC")
#train_svm(c=1, class_weight='balanced', name="penalty_1_balanced_SVC")