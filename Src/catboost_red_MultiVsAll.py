import os

print(not((os.path.exists("./catboost modele i wyniki/CPU RED MultiVsAll")) and (os.path.exists("./catboost RED modele i wyniki/GPU MultiVsAll"))))


if not((os.path.exists("./catboost modele i wyniki/CPU RED MultiVsAll")) and (os.path.exists("./catboost RED modele i wyniki/GPU MultiVsAll"))):
    import catboost_red_MultiVsAll_modeling
import catboost_red_MultiVsAll_testing

if not((os.path.exists("./catboost modele i wyniki/CPU RED MultiVsAll verification")) and (os.path.exists("./catboost RED modele i wyniki/GPU basic verification"))):
    import catboost_red_MultiVsAll_modeling_verification
import catboost_red_MultiVsAll_testing_verification

