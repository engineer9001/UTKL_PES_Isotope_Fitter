#This is supposed to be an improvement on the previously-implimented way of specifying
#the name of PES-isotopes and their respective mean lifetimes. This new implimentation will boil
#down to tossing everything in a big dictionary where the key will be the name fo the isotope,
#and the value will be the mean lifetime in untis of seconds

#ADDITIOANL ISOTOPES SHOULD BE ADDDED FOLLOWNG THIS FORMAT
from numpy import log
ln2 = log(2)

tau_Cu58 = 3.204/ln2
tau_Cu59 = 89.5/ln2
tau_Cu60 = 1422/ln2
tau_Cu62 = 9.672*60/ln2
tau_Zn60 = 2.38*60/ln2
tau_Zn61 = 89.1/ln2
tau_Zn63 = 38.47*60/ln2
tau_Co54m = 88.8/ln2

Isotopes_Lifetimes_Dict = {
    "Cu58": tau_Cu58,
    "Cu59": tau_Cu59,
    "Cu60": tau_Cu60,
    "Cu62": tau_Cu62,
    "Zn60": tau_Zn60,
    "Zn61": tau_Zn61,
    "Zn63": tau_Zn63,
    "Co54m": tau_Co54m
}
