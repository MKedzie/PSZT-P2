import wines_import
import catboost
import numpy
from pandas.core.frame import  DataFrame
from sklearn import metrics

red_quality_predicted_CPU: numpy.ndarray
red_quality_test: DataFrame


red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)
red_model_CPU_basic = catboost.CatBoostClassifier()
red_model_CPU_basic.load_model("./catboost modele i wyniki/CPU RED basic verification")
red_model_GPU_basic = catboost.CatBoostClassifier()
red_model_GPU_basic.load_model("./catboost modele i wyniki/GPU RED basic verification")
messages_file = open("./catboost modele i wyniki/models RED basic results verification", mode="w+")
red_quality_predicted_CPU = red_model_CPU_basic.predict(red_data_test)
red_quality_predicted_GPU = red_model_GPU_basic.predict(red_data_test)
print(red_quality_test.to_numpy())
print(red_quality_predicted_CPU)
print(red_quality_predicted_GPU)

messages_file.write("RMSE CPU\n")
messages_file.write(str(numpy.abs((red_quality_test-red_quality_predicted_CPU)).mean()))
messages_file.write("RMSE GPU\n")
messages_file.write(str(numpy.abs((red_quality_test-red_quality_predicted_GPU)).mean()))

messages_file.write("Catboost uczony CPU wyniki\n")
messages_file.write(str(metrics.classification_report(red_quality_test, red_quality_predicted_CPU,zero_division=0)))
messages_file.write(str(metrics.confusion_matrix(red_quality_test, red_quality_predicted_CPU)))
messages_file.write("Catboost uczony GPU wyniki\n")
messages_file.write(str(metrics.classification_report(red_quality_test, red_quality_predicted_GPU,zero_division=0)))
messages_file.write(str(metrics.confusion_matrix(red_quality_test, red_quality_predicted_GPU)))