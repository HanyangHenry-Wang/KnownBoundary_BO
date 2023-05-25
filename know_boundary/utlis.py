from botorch.test_functions import Ackley,Levy,Beale,Branin,Hartmann,Rosenbrock
import torch
from botorch.utils.sampling import draw_sobol_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def get_initial_points(bounds,num,device,dtype,seed=0):
    
        train_x = draw_sobol_samples(
        bounds=bounds, n=num, q=1,seed=seed).reshape(num,-1).to(device, dtype=dtype)
        
        return train_x
    
    
class Trans_function:
    def __init__(self,fun,fstar,min=True):
        self.fun = fun
        self.fstar = fstar
        self.min = min
        
        
    def __call__(self, X):
        
        if self.min:
            y = self.fun(X)-self.fstar
        
        return y