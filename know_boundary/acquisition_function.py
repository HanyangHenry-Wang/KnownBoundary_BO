import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from scipy import special
from scipy.stats import norm

##################### GP acquisition function ########################################################
def EI(X,dim,f_best,model): # X is a 2-dimensional array because we will use it in scipy.minimize

  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)
  
  var[var<10**(-12)]=10**(-12)

  z = (f_best - mean)/np.sqrt(var)        
  out=(f_best - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

  return out.ravel()  #make the shape to be 1 dimensional



def EI_acquisition_opt(model,bounds,f_best): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = EI(init_X,dim,f_best,model)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -EI(X=x,dim=dim,f_best=f_best,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x  
    AF_temp = EI(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,model=model)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next

def MES(X,dim,fstar,model): 
  
  X = X.reshape(-1,dim)
  mean,var = model.predict(X,include_likelihood=False)

  var[var<10**(-12)]=10**(-12)
  gamma = -(fstar-mean)/np.sqrt(var)  

  cdf_part = norm.cdf(gamma)
  out = (gamma*norm.pdf(gamma))/(2*cdf_part)-np.log(cdf_part)

  return out.ravel() 



def MES_acquisition_opt(model,bounds,fstar): #bound should an array of size dim*2

  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = MES(init_X,dim,fstar,model)
        
    x0=init_X[np.argmax(value_holder)]


    res = minimize(lambda x: -MES(X=x,dim=dim,fstar=fstar,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

    X_temp =  res.x
    AF_temp = MES(X=np.array(X_temp).reshape(-1,1),dim=dim,fstar=fstar,model=model)
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]


  return X_next


##################### log GP acquisition function ########################################################

def Warped_TEI2(X,dim,f_best,c,f_mean,model): # X is a 2-dimensional array because we will use it in scipy.minimize


  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)  
  var[var<10**(-12)]=10**(-12)
  sigma = np.sqrt(var)
  mu = mean+f_mean

  C = c+f_best
  
  part1 = C*norm.cdf((np.log(C)-mu)/sigma)-np.exp(mu+sigma**2/2)*norm.cdf((np.log(C)-mu-sigma**2)/sigma)
  
  C = c
  part2 = C*norm.cdf((np.log(C)-mu)/sigma)-np.exp(mu+sigma**2/2)*norm.cdf((np.log(C)-mu-sigma**2)/sigma)
  
  out_temp = part1-part2
  
  part3 = f_best*norm.cdf(  (np.log(c)-mu) /sigma ) 
  
  out = out_temp+part3
  

  return out.ravel()  #make the shape to be 1 dimensional



def Warped_TEI2_acquisition_opt(model,bounds,f_best,c,f_mean): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder =  Warped_TEI2(init_X,dim,f_best,c,f_mean,model)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -Warped_TEI2(X=x,dim=dim,f_best=f_best,c=c,f_mean=f_mean,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

    X_temp =   res.x  
    AF_temp = Warped_TEI2(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,c=c,f_mean=f_mean,model=model)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next