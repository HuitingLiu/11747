import math
import locale
import re
from word2number import w2n
from fractions import Fraction
locale.setlocale( locale.LC_NUMERIC, 'en_US.UTF-8' ) 

ordinal = {'first':1, 'second':2, 'third':3, 'forth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'eighth':8, 'ninth':9, 'tenth':10}
argsNum = {'ADD':2, 'SUBSTRACT':2, 'MULTIPLY':2, 'DIVIDE':2, 'POWER':2, 'LOG':1, 'SQRT':1, 
           'SIN':1, 'COS':1, 'TAN':1, 'RADIUS2DEGREE':1, 'DEGREE2RADIUS':1, 'FACTORIAL':1, 'CHOOSE':2}


def ADD(x1, x2):
    if type(x1) == float and type(x2) == float:
        return (True, x1 + x2)
    return (False, None)

def SUBSTRACT(x1, x2):
    if type(x1) == float and type(x2) == float:
        return (True, x1 - x2)
    return (False, None)

def MULTIPLY(x1, x2):
    if type(x1) == float and type(x2) == float:
        return (True, x1 * x2)
    return (False, None)
    
def DIVIDE(x1, x2):
    if type(x1) == float and type(x2) == float and x2 != 0:
        return (True, x1 / x2)
    return (False, None)
    
def POWER(x1, x2):
    if type(x1) == float and type(x2) == float:
        return (True, x1 ** x2)
    return (False, None)

def LOG(x1):
    try:
        return (True, math.log(x1))
    except:
        return (False, None)

def SQRT(x1):
    try:
        return (True, math.sqrt(x1))
    except:
        return (False, None)
    
def SIN(x1):
    try:
        return (True, math.sin(x1))
    except:
        return (False, None)
    
def COS(x1):
    try:
        return (True, math.cos(x1))
    except:
        return (False, None)
    
def TAN(x1):
    try:
        return (True, math.cos(x1))
    except:
        return (False, None)

def RADIUS2DEGREE(x1):
    try:
        return (True, math.degrees(x1))
    except:
        return (False, None)
    
def DEGREE2RADIUS(x1):
    try:
        return (True, math.radians(x1))
    except:
        return (False, None)
    
def FACTORIAL(x1):
    try:
        return (True, math.factorial(x1))
    except:
        return (False, None)
    
def CHOOSE(x1, x2):
    try:
        a = math.factorial(x1)
        b = math.factorial(x2)
        if x2 > x1:
            return (False, None)
        return (True, a / b / math.factorial(x1 - x2))
    except:
        return (False, None)

def STR2FLOAT(x1):
    try:
        return (True, locale.atof(x1))
    except: pass
    
    tmp = re.sub('[^0-9/%]',' ', x1)
    
    try:
        return (True, float(Fraction(tmp)))
    except: pass
    
    try:
        return (True, float(tmp.strip('%'))/100)
    except: pass
    
    try:
        return (True, locale.atof(temp))
    except: pass
    
    try:
        return (True, float(w2n.word_to_num(x1)))
    except: pass
    
    try:
        return (True, float(ordinal[x1]))
    except: pass
    return (False, None)
    
def FLOAT2STR(x1):
    if type(x1) == float:
        return (True, str(x1))
    return (False, None)


def CHECK(x1):
    pass