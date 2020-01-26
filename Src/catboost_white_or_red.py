import wines_import
import pandas.core.frame
import numpy
import catboost
from sklearn import metrics
import os

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

if not (os.path.exists("./catboost modele i wyniki/COLOR")):
    color_classifier = catboost.CatBoostClassifier(random_seed=42)
    color_classifier.fit(color_data_training,color_color_training)
else:
    color_classifier = catboost.CatBoostClassifier().load_model("./catboost modele i wyniki/COLOR")
color_classifier.save_model("./catboost modele i wyniki/COLOR")
color_predict = color_classifier.predict(color_data_test)


print(metrics.classification_report(color_color_test,color_predict,zero_division=0))