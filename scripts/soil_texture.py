import numpy as np

# Soil texture function
def soil_texture(sand,clay,silt):
 # function to calculate soil texture based on sand, clay, and silt percetage
 temp=np.ones(sand.shape)*(-9999)
 mask = (sand == -9999) | ( (clay == -9999) | (silt == -9999) )

 mapping = {
 'sand':1,
 'loamy_sand':2,
 'sandy_loam':3,
 'silt_loam':4,
 'silt':5,
 'loam':6,
 'sandy_clay_loam':7,
 'silty_clay_loam':8,
 'clay_loam':9,
 'sandy_clay':10,
 'silty_clay':11,
 'clay':12
 } 

 # Sand
 ma0 = ((silt+1.5*clay)<15)    
 temp[ma0] = mapping['sand']

 # Loamy sand
 ma1 = ((silt+1.5*clay)>=15) & ((silt+2*clay)<30)
 temp[ma1] = mapping['loamy_sand']
 
 # Sandy loam
 m21 = ((clay)>=7) & ((clay)<20) & ((sand)>52) & ((silt+2*clay)>=30)
 m22= ((clay<7) & (silt<50) & ((silt+2*clay)>=30))
 ma2 = m21|m22
 temp[ma2] = mapping['sandy_loam']
 
 # Loam
 m31 = (clay>=7) & (clay<27)
 m32 = (silt>=28) & (silt<50)
 m33 = (sand<=52)
 ma3 = m31 & m32 & m33
 temp[ma3] = mapping['loam']
 
 # Silt loam
 m41 = (silt>=50) & ((clay>=12) & (clay<27))
 m42 = ((silt>=50) & (silt<80)) & (clay<12)
 ma4 = m41|m42
 temp[ma4] = mapping['silt_loam']
 
 # Silt
 ma5 = (silt>=80) & (clay<12)
 temp[ma5]= mapping['silt']
 
 # Sandy clay loam
 m61 = ((clay>=20) & (clay<35))
 ma6 = m61 & (silt<28) & (sand>45)
 temp[ma6] = mapping['sandy_clay_loam']
 
 # Clay loam
 m71 = (clay>=27) & (clay<40)
 m72 = (sand>20) & (sand<=45)
 ma7 = m71 & m72
 temp[ma7] = mapping['clay_loam']
 
 # Silty clay loam
 ma8 = ((clay>=27) & (clay<40)) & (sand<=20)
 temp[ma8] = mapping['silty_clay_loam']
 
 # Sandy clay
 ma9 = (clay>=35) & (sand>45)
 temp[ma9] = mapping['sandy_clay']
 
 # Silty clay
 ma10 = (clay>=40) & (silt>=40)
 temp[ma10] = mapping['silty_clay']
 
 # Clay
 ma11 = (clay>=40) & (sand<=45) & (silt<40)
 temp[ma11] = mapping['clay']

 temp[mask] = -9999 
 return temp


