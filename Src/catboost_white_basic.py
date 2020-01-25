import os

print(not((os.path.exists("./catboost modele i wyniki/CPU WHITE basic")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE basic"))))


if not((os.path.exists("./catboost modele i wyniki/CPU WHITE basic")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE basic"))):
    import catboost_white_basic_modeling
import catboost_white_basic_testing

if not((os.path.exists("./catboost modele i wyniki/CPU WHITE basic verification")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE basic verification"))):
    import catboost_white_basic_modeling_verification
import catboost_white_basic_testing_verification

