import os

print(not((os.path.exists("./catboost modele i wyniki/CPU RED basic")) and (os.path.exists("./catboost modele i wyniki/GPU RED basic"))))


if not((os.path.exists("./catboost modele i wyniki/CPU RED basic")) and (os.path.exists("./catboost modele i wyniki/GPU RED basic"))):
    import catboost_red_basic_modeling
import catboost_red_basic_testing

if not((os.path.exists("./catboost modele i wyniki/CPU RED basic verification")) and (os.path.exists("./catboost modele i wyniki/GPU RED basic verification"))):
    import catboost_red_basic_modeling_verification
import catboost_red_basic_testing_verification

