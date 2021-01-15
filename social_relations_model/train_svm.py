import pandas
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pandas.read_csv('data/train_model_data.csv')
X = data.drop(columns=['group_label'])
y = data['group_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("writing to outfile")
outfile = "data/social_relations.model"
pickle.dump(svc, open(outfile, "wb"))
print("finished")