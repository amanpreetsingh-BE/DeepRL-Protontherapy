import numpy as np
import math

def cdirect(CDT,u,v,w ):

  phi = 2*math.pi*np.random.rand()     

  CDF = math.cos(phi)
  SDF = math.sin(phi)
  norm = math.sqrt(u**2+v**2+w**2)
  u = u/norm
  v = v/norm
  w = w/norm
  up = u
  DXY  = u*u + v*v
        
  if (DXY >= 10**(-10)):
    SDT = math.sqrt((1.0-CDT*CDT)/DXY)
    u = u*CDT+SDT*(up*w*CDF-v*SDF)
    v = v*CDT+SDT*(v*w*CDF+up*SDF)
    w = w*CDT-DXY*SDT*CDF
  else:
    SDT = math.sqrt(1.0-CDT*CDT)
    u = SDT*CDF
    v = SDT*SDF
    w = CDT
    
  return np.array([u,v,w])

