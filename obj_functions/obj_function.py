import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy import genfromtxt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch


class XGBoost:
    def __init__(self,seed=1):
        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [0.,10.],  # alpha
                                  [0.,10.],# gamma 
                                  [5.,15.], #max_depth
                                  [1.,20.],  #min_child_weight
                                  [0.5,1.],  #subsample
                                  [0.1,1] #colsample
                                 ]).T)
            
        self.dim = 6
        self.fstar = 100

        self.seed= seed
        self.data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)
        
    def __call__(self, X): # this is actually a Branin function
        
        X = X.numpy().reshape(6,)

        
        alpha,gamma,max_depth,min_child_weight,subsample,colsample=X[0],X[1],X[2],X[3],X[4],X[5]
        
        #data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)

        outputs = self.data[:,3]
        inputs = self.data[:,0:3]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.85, random_state=self.seed)
        y_train1 = y_train1-1
        
        reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                       min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'binary:logistic', booster='gbtree',eval_metric='logloss',silent=None)
        score = np.array(cross_val_score(reg, X=X_train1, y=y_train1).mean())
      
        return torch.tensor([score*100])