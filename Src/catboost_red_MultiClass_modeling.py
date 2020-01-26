import wines_import
import catboost
import time
import os

red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)

if(os.path.exists("./catboost modele i wyniki/CPU RED MultiClass")):
    os.remove("./catboost modele i wyniki/CPU RED MultiClass")
if(os.path.exists("./catboost modele i wyniki/GPU RED MultiClass")):
    os.remove("./catboost modele i wyniki/GPU RED MultiClass")


messages_file = open("./catboost modele i wyniki/models RED MultiClass", mode="w+")
model_red = catboost.CatBoostClassifier(task_type="CPU", random_seed=42, objective="MultiClass",iterations=2000,reg_lambda=0.1)
time_before = time.time()
model_red.fit(red_data_training, red_quality_training)
time_after = time.time()
model_red.save_model("./catboost modele i wyniki/CPU RED MultiClass")
messages_file.write("Uczenie na CPU wina czerwone trwalo:\n")
messages_file.write(str(time_after - time_before))
model_red_GPU = catboost.CatBoostClassifier(task_type="GPU", random_seed=42, objective="MultiClass",iterations=2000,reg_lambda=0.1)
time_before = time.time()
model_red_GPU.fit(red_data_training, red_quality_training)
time_after = time.time()
model_red_GPU.save_model("./catboost modele i wyniki/GPU RED MultiClass")
messages_file.write("\nUczenie na GPU wina czerwone trwalo:\n")
messages_file.write(str(time_after - time_before))
