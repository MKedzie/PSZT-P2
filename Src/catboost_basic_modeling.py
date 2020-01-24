import wines_import
import catboost
import time

red_data_training, red_data_test, red_quality_training, red_quality_test,white_data_training,white_data_test, white_quality_training, white_quality_test = wines_import.read_data(True)

plik = open("./uczenie podstawowe",mode="w+")
model_red = catboost.CatBoostClassifier(task_type="CPU")
print(red_data_training)
time_before = time.time()
model_red.fit(red_data_training, red_quality_training)
time_after = time.time()
model_red.save_model("./model CPU")
plik.write("Uczenie na CPU wina czerwone trwalo:\n")
plik.write(str(time_after-time_before))
model_red_GPU = catboost.CatBoostClassifier(task_type="GPU")
time_before = time.time()
model_red_GPU.fit(red_data_training,red_quality_training)
time_after = time.time()
model_red_GPU.save_model("./model GPU")
plik.write("Uczenie na GPU wina czerwone trwalo:\n")
plik.write(str(time_after-time_before))

#print(model_red)
#expected_red_quality = red_quality_test
#predicted_red_quality = model_red.predict(red_data_test)
