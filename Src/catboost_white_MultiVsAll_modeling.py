import wines_import
import catboost
import time
import os

red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)

if(os.path.exists("./catboost modele i wyniki/CPU WHITE MultiVsAll")):
    os.remove("./catboost modele i wyniki/CPU WHITE MultiVsAll")
if(os.path.exists("./catboost modele i wyniki/GPU WHITE MultiVsAll")):
    os.remove("./catboost modele i wyniki/GPU WHITE MultiVsAll")


messages_file = open("./catboost modele i wyniki/models WHITE MultiVsAll", mode="w+")
model_white = catboost.CatBoostClassifier(task_type="CPU", random_seed=42, objective="MultiClassOneVsAll",iterations=2000)
time_before = time.time()
model_white.fit(white_data_training, white_quality_training)
time_after = time.time()
model_white.save_model("./catboost modele i wyniki/CPU WHITE MultiVsAll")
messages_file.write("Uczenie na CPU wina biale trwalo:\n")
messages_file.write(str(time_after - time_before))
model_white_GPU = catboost.CatBoostClassifier(task_type="GPU", random_seed=42, objective="MultiClassOneVsAll",iterations=2000)
time_before = time.time()
model_white_GPU.fit(white_data_training, white_quality_training)
time_after = time.time()
model_white_GPU.save_model("./catboost modele i wyniki/GPU WHITE MultiVsAll")
messages_file.write("\nUczenie na GPU wina biale trwalo:\n")
messages_file.write(str(time_after - time_before))
