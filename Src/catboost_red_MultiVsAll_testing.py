import wines_import
import catboost
import numpy
from pandas.core.frame import  DataFrame
from sklearn import metrics

red_quality_predicted_CPU: numpy.ndarray
red_quality_test: DataFrame


red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)
red_model_CPU_basic = catboost.CatBoostClassifier()
red_model_CPU_basic.load_model("./catboost modele i wyniki/CPU RED MultiVsAll")
red_model_GPU_basic = catboost.CatBoostClassifier()
red_model_GPU_basic.load_model("./catboost modele i wyniki/GPU RED MultiVsAll")
messages_file = open("./catboost modele i wyniki/models RED MultiVsAll results", mode="w+")
red_quality_predicted_CPU = red_model_CPU_basic.predict(red_data_test)
red_quality_predicted_GPU = red_model_GPU_basic.predict(red_data_test)
print(red_quality_test.to_numpy())
print(red_quality_predicted_CPU)
print(red_quality_predicted_GPU)

print("RMSE CPU")
print(numpy.sqrt((red_quality_test-red_quality_predicted_CPU)**2).mean())
print("RMSE GPU")
print(numpy.sqrt((red_quality_test-red_quality_predicted_GPU)**2).mean())

print("Catboost uczony CPU wyniki\n")
print(metrics.classification_report(red_quality_test, red_quality_predicted_CPU,zero_division=0))
print("Catboost uczony GPU wyniki\n")
print(metrics.classification_report(red_quality_test, red_quality_predicted_GPU,zero_division=0))

