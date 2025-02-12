# Creation of the following box should be programmed in a way to take the next box with the specific isotopes. Likely create a class that can return a single in the form of
# def Cu58(t,A):
#     tau_Cu58 = 3.204/ln2
#     return A*e**(-t/tau_Cu58)
# 
# Probably have the tau or half life be an initialization parameter and the actual function above be a subroutine.

#-----------------------------------------------------------------------------------------

import numpy as np #use np.exp it's way faster

ln2 = 0.6391

#Constant for background (presumably)
constant = 42.447360677913636



#Converting half-lives to mean lifetimes
tau_Cu58 = 3.204/ln2
tau_Cu59 = 89.5/ln2
tau_Cu60 = 1422/ln2
tau_Cu62 = 9.672*60/ln2
tau_Zn60 = 2.38*60/ln2
tau_Zn61 = 89.1/ln2
tau_Zn63 = 38.47*60/ln2
tau_Co54m = 88.8/ln2

def single(t,A,tau1):
    return A*np.exp(-t/tau1) + constant

def double(t,A,tau1,B,tau2):
    return A*np.exp(-t/tau1) + B*np.exp(-t/tau2)  + constant #C*np.exp(-t/tau3)

def double_fit2constant(t,A,tau1,B,tau2,C):
    return A*np.exp(-t/tau1) + B*np.exp(-t/tau2) + C

def triple(t,A,tau1,B,tau2,C,tau3):
    return A*np.exp(-t/tau1) + B*np.exp(-t/tau2)  + C*np.exp(-t/tau3) + constant

def double_Cu58_Cu62(t,A,B,C):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu62) + C

def triple_Cu58_Cu62_Cu59(t,A,B,C):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu62) + C*np.exp(-t/tau_Cu59)

def triple_Cu58_Cu62_Zn60(t,A,B,C):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu62) + C*np.exp(-t/tau_Zn60)

def quad_Cu58_Cu62_Cu59_Zn60(t,A,B,C,D):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu62) + C*np.exp(-t/tau_Cu59) + D*np.exp(-t/tau_Zn60)

def quad_Cu58_Cu60_Cu59_Cu62(t,A,B,C,D):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu60) + C*np.exp(-t/tau_Cu59) + D*np.exp(-t/tau_Cu62)

def all_Cu58_Cu62_Cu59_Cu60_Zn60_Zn61_Zn63_Co54m(t,A,B,C,D,E,F,G,H):
    return A*np.exp(-t/tau_Cu58) + B*np.exp(-t/tau_Cu62) + C*np.exp(-t/tau_Cu59) + D*np.exp(-t/tau_Cu60) + E*np.exp(-t/tau_Zn60) + F*np.exp(-t/tau_Zn61) + G*np.exp(-t/tau_Zn63) + H*np.exp(-t/tau_Co54m)

def Cu58(t,A):
    return A*np.exp(-t/tau_Cu58)

def Cu59(t,A):
    return A*np.exp(-t/tau_Cu59)

def Cu60(t,A):
    return A*np.exp(-t/tau_Cu60)

def Cu62(t,A):
    return A*np.exp(-t/tau_Cu62)

def Zn60(t,A):
    return A*np.exp(-t/tau_Zn60)

def Zn61(t,A):
    return A*np.exp(-t/tau_Zn61)

def Zn63(t,A):
    return A*np.exp(-t/tau_Zn63)

def Co54m(t,A):
    return A*np.exp(-t/tau_Co54m)

#DON'T KNOW WHERE THESE SHOULD GO SO THEY'RE HERE NOW
# Spill times/run lengths need to be adjusted

spillTimes = [[52.60406546037813, 52.70576546037813],
 [7.658639091560348, 7.760139091560347],
 [21.685239651808505, 21.786839651808506],
 [62.611831181721065, 62.71383118172106],
 [10.560941430694248, 10.662941430694248],
 [68.50573713055009, 68.60743713055008],
 [52.370972189816804, 52.4725721898168],
 [13.310021040987447, 13.411821040987448],
 [69.17295778706057, 69.27475778706057],
 [23.0682897979999, 23.1698897979999],
 [39.9248759449842, 40.0264759449842], #nickel
 [76.49762062100974, 76.59942062100974], #copper
 [100.57416844934949, 100.67546844934948]]

runLengths = [180,180,180,1200,1200,900,900,900,300,1200,600,600,900] #nickel index 10, copper index 11