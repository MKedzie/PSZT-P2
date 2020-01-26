import wines_import
import pandas.core.frame
import numpy
import time
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)

red_data_training : pandas.core.frame.DataFrame
red_data_test : pandas.core.frame.DataFrame
red_quality_training : pandas.core.frame.DataFrame
red_quality_test : pandas.core.frame.DataFrame
white_data_training : pandas.core.frame.DataFrame
white_data_test : pandas.core.frame.DataFrame
white_quality_training : pandas.core.frame.DataFrame
white_quality_test : pandas.core.frame.DataFrame

red_color_training = numpy.full(red_quality_training.shape,fill_value="red")
red_color_training = pandas.core.frame.DataFrame(data=red_color_training,columns=['color'])
red_color_test = numpy.full(red_quality_test.shape,fill_value="red")
red_color_test = pandas.core.frame.DataFrame(data=red_color_test,columns=["color"])


white_color_training = numpy.full(white_quality_training.shape,fill_value="white")
white_color_training = pandas.core.frame.DataFrame(white_color_training,columns=["color"])
white_color_test = numpy.full(white_quality_test.shape,fill_value="white")
white_color_test = pandas.core.frame.DataFrame(white_color_test,columns=["color"])

color_color_training = pandas.concat([red_color_training,white_color_training])
color_color_test = pandas.concat([red_color_test,white_color_test])

color_data_training = pandas.concat([red_data_training,white_data_training])
color_data_test = pandas.concat([red_data_test,white_data_test])


time_before=time.time()

reg = LogisticRegression(solver='saga', random_state=42, max_iter=2000, multi_class='auto')
reg.fit(color_data_training, color_color_training)
y_pred = reg.predict(color_data_test)
time_after=time.time()

print("Regression coefficient is ", reg.coef_)
#classification report
print(metrics.classification_report(color_color_test, y_pred))
Conf_Mat = metrics.confusion_matrix(color_color_test, y_pred)
print("The confusion matrix is\n", Conf_Mat)
print("Accuracy is ",metrics.accuracy_score(color_color_test, y_pred))
print("Czas wykonania: ",round(time_after-time_before,3)," sekund")