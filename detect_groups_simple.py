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