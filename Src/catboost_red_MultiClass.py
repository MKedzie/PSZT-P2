import os

print(not((os.path.exists("./catboost modele i wyniki/CPU RED MultiClass")) and (os.path.exists("./catboost modele i wyniki/GPU RED MultiClass"))))


if not((os.path.exists("./catboost modele i wyniki/CPU RED MultiClass")) and (os.path.exists("./catboost modele i wyniki/GPU RED MultiClass"))):
    import catboost_red_MultiClass_modeling
import catboost_red_MultiClass_testing

if not((os.path.exists("./catboost modele i wyniki/CPU RED MultiClass verification")) and (os.path.exists("./catboost modele i wyniki/GPU RED MultiClass verification"))):
    import catboost_red_MultiClass_modeling_verification
import catboost_red_MultiClass_testing_verification

