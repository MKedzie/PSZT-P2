#import needed library
from  wines_import import read_data
import pandas as pd
#import linear model and train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#Read the data file
data=pd.read_csv('../Data/winequality-white.csv')
red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = read_data(True)

reg = LogisticRegression()
reg.fit(white_data_training, white_quality_training)
y_pred = reg.predict(white_data_test)
print("Regression coefficient is ", reg.coef_)
#classification report
print(metrics.classification_report(white_quality_test, y_pred))
Conf_Mat = metrics.confusion_matrix(white_quality_test, y_pred)
print("The confusion matrix is\n", Conf_Mat)
print("Accuracy is ",metrics.accuracy_score(white_quality_test, y_pred))
