import os

print(not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiClass")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE MultiClass"))))


if not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiClass")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE MultiClass"))):
    import catboost_white_MultiClass_modeling
import catboost_white_MultiClass_testing

if not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiClass verification")) and (os.path.exists("./catboost modele i wyniki/GPU WHITE MultiClass verification"))):
    import catboost_white_MultiClass_modeling_verification
import catboost_white_MultiClass_testing_verification

