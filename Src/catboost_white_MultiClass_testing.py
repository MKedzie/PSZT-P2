import wines_import
import catboost
import numpy
from pandas.core.frame import  DataFrame
from sklearn import metrics

white_quality_predicted_CPU: numpy.ndarray
white_quality_test: DataFrame


red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)

white_model_CPU_basic = catboost.CatBoostClassifier()
white_model_CPU_basic.load_model("./catboost modele i wyniki/CPU WHITE MultiClass")
white_model_GPU_basic = catboost.CatBoostClassifier()
white_model_GPU_basic.load_model("./catboost modele i wyniki/GPU WHITE MultiClass")
messages_file = open("./catboost modele i wyniki/models WHITE MultiClass results", mode="w+")
white_quality_predicted_CPU = white_model_CPU_basic.predict(white_data_test)
white_quality_predicted_GPU = white_model_GPU_basic.predict(white_data_test)


messages_file.write("\nRMSE CPU\n")
messages_file.write(str(numpy.abs((white_quality_test-white_quality_predicted_CPU)).mean()))
messages_file.write("\nRMSE GPU\n")
messages_file.write(str(numpy.abs((white_quality_test-white_quality_predicted_GPU)).mean()))

messages_file.write("\nCatboost uczony CPU wyniki\n")
messages_file.write(str(metrics.classification_report(white_quality_test, white_quality_predicted_CPU,zero_division=0)))
messages_file.write(str(metrics.confusion_matrix(white_quality_test, white_quality_predicted_CPU)))
messages_file.write("\nCatboost uczony GPU wyniki\n")
messages_file.write(str(metrics.classification_report(white_quality_test, white_quality_predicted_GPU,zero_division=0)))
messages_file.write(str(metrics.confusion_matrix(white_quality_test, white_quality_predicted_GPU)))
