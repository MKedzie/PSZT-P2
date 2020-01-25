import wines_import
import catboost
import numpy
from pandas.core.frame import  DataFrame
from sklearn import metrics

white_quality_predicted_CPU: numpy.ndarray
white_quality_test: DataFrame


red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(True)

white_model_CPU_basic = catboost.CatBoostClassifier()
white_model_CPU_basic.load_model("./catboost modele i wyniki/CPU WHITE MultiClass")
white_model_GPU_basic = catboost.CatBoostClassifier()
white_model_GPU_basic.load_model("./catboost modele i wyniki/GPU WHITE MultiClass")
messages_file = open("./catboost modele i wyniki/models WHITE MultiClass results", mode="w+")
white_quality_predicted_CPU = white_model_CPU_basic.predict(white_data_test)
white_quality_predicted_GPU = white_model_GPU_basic.predict(white_data_test)
print(white_quality_test.to_numpy())
print(white_quality_predicted_CPU)
print(white_quality_predicted_GPU)

print("RMSE CPU")
print(numpy.sqrt((white_quality_test-white_quality_predicted_CPU)**2).mean())
print("RMSE GPU")
print(numpy.sqrt((white_quality_test-white_quality_predicted_GPU)**2).mean())

print("Catboost uczony CPU wyniki\n")
print(metrics.classification_report(white_quality_test, white_quality_predicted_CPU,zero_division=0))
print("Catboost uczony GPU wyniki\n")
print(metrics.classification_report(white_quality_test, white_quality_predicted_GPU,zero_division=0))

