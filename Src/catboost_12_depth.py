import os
import catboost
from sklearn import metrics

import wines_import

if(os.path.exists("./catboost modele i wyniki/12 depth/CPU RED basic") and os.path.exists("./catboost modele i wyniki/12 depth/GPU RED basic")):

    model_red = catboost.CatBoostClassifier()
    model_red_GPU = catboost.CatBoostClassifier()
    model_red.load_model("./catboost modele i wyniki/12 depth/CPU RED basic")
    model_red_GPU.load_model("./catboost modele i wyniki/12 depth/GPU RED basic")
    if model_red.tree_count_>10 and model_red_GPU.tree_count_>10:
        os.remove("./catboost modele i wyniki/12 depth/CPU RED basic")
        os.remove("./catboost modele i wyniki/12 depth/GPU RED basic")
        model_red.shrink(model_red.tree_count_,model_red.tree_count_-2)
        model_red_GPU.shrink(model_red_GPU.tree_count_, model_red_GPU.tree_count_-2)
        model_red.save_model("./catboost modele i wyniki/12 depth/CPU RED basic")
        model_red_GPU.save_model("./catboost modele i wyniki/12 depth/GPU RED basic")
    red_data_training, red_data_test, red_quality_training, red_quality_test, white_data_training, white_data_test, white_quality_training, white_quality_test = wines_import.read_data(False)

    red_quality_predicted_CPU = model_red.predict(red_data_test)
    red_quality_predicted_GPU = model_red_GPU.predict(red_data_test)
    print("\nCatboost uczony CPU wyniki\n")
    print(metrics.classification_report(red_quality_test, red_quality_predicted_CPU, zero_division=0))
    print(metrics.confusion_matrix(red_quality_test, red_quality_predicted_CPU))
    print("\nCatboost uczony GPU wyniki\n")
    print(metrics.classification_report(red_quality_test, red_quality_predicted_GPU, zero_division=0))
    print(metrics.confusion_matrix(red_quality_test, red_quality_predicted_GPU))