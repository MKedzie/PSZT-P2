import os

print(not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiVsAll")) and (os.path.exists("./catboost WHITE modele i wyniki/GPU MultiVsAll"))))


if not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiVsAll")) and (os.path.exists("./catboost WHITE modele i wyniki/GPU MultiVsAll"))):
    import catboost_white_MultiVsAll_modeling
import catboost_white_MultiVsAll_testing

if not((os.path.exists("./catboost modele i wyniki/CPU WHITE MultiVsAll verification")) and (os.path.exists("./catboost WHITE modele i wyniki/GPU basic verification"))):
    import catboost_white_MultiVsAll_modeling_verification
import catboost_white_MultiVsAll_testing_verification

