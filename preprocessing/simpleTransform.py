#!/usr/bin/python
# PyElly - scripting tool for analyzing natural language
#
# simpleTransform.py : 15mar2016 CPM
# ------------------------------------------------------------------------------
# Copyright (c) 2013, Clinton Prentiss Mah
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

"""
basic class for normalizing and identifying various kinds of entities

This defines the rewriteNumber() method, which should NOT be called by
any SimpleTransform subclass because this action has to be controlled
by the ellyConfiguration module. The arrangement is awkward, but needed
to avoid circular importing of modules.
"""

from . import ellyChar

N = 20       # upper limit on substring for match

ALSO   = [ "-" , "/" , "'", "%" ]  # accept in sublist gotten also
DASH  = '-'
COMMA  = ',' # number divider
DOT  = '.' # decimal point
PERCENT  = '%'
SLASH  = '/'

# general recognition support for spelled out numbers

Card = {
    'one':1 , 'two':2   , 'three':3 , 'four':4 , 'five':5 ,
    'six':6 , 'seven':7 , 'eight':8 , 'nine':9 , 'ten':10 ,
    'eleven':11  , 'twelve':12    , 'thirteen':13 , 'fourteen':14 , 'fifteen':15 ,
    'sixteen':16 , 'seventeen':17 , 'eighteen':18 , 'nineteen':19 ,
    'twenty':20 , 'thirty':30  , 'forty':40  , 'fifty':50 ,
    'sixty':60  , 'seventy':70 , 'eighty':80 , 'ninety':90
}

Ord  = {
    'first':1 , 'second':2  , 'third':3  , 'fourth':4 , 'fifth':5  ,
    'sixth':6 , 'seventh':7 , 'eighth':8 , 'nineth':9 , 'ninth':9 , 'tenth':10 ,
    'eleventh':11  , 'twelfth':12     , 'thirteenth':13 , 'fourteenth':14 , 'fifteenth':15 ,
    'sixteenth':16 , 'seventeenth':17 , 'eighteenth':18 , 'nineteenth':19 ,
    'twentieth':20 , 'thirtieth':30  , 'fortieth':40  , 'fiftieth':50 ,
    'sixtieth':60  , 'seventieth':70 , 'eightieth':80 , 'ninetieth':90
}

Magn = {
     'percent':0.01 , 'percents':0.01 , '%':0.01 , 'hundred':100 , 'thousand':1000 , 'million':1000000 ,
     'billion':1000000000 , 'trillion':1000000000000 ,
     'hundredth':100 , 'thousandth':1000 , 'millionth':1000000 ,
     'billionth':1000000000 , 'trillionth':1000000000000
}

class SimpleTransform(object):

    """
    base class to match strings for text entity recognition

    attributes:
        string - substring for match
    """

    def __init__ ( self ):

        """
        initialization

        arguments:
            self
        """

        self.string = None
        self.reset()

    def reset ( self ):

        """
        restore to initial state

        arguments:
            self
        """

        self.string = ''

    def get ( self , ts , n=N ):

        """
        get normalized substring in lower case for subsequent comparisons

        arguments:
            self -
            ts   - list of chars to get substring from
            n    - limit on count of chars to get

        returns:
            count of chars scanned for substring
        """

        sl = [ ]                          # char sublist to be matched
#       print 'ts=' , ts
        lts = len(ts)
        if lts == 0:
            return 0                      # no chars to scan
        lm = lts if lts < n else n
#       print 'lm=' , lm
        i = 0
        c = ''
        meeted = set()
        while i < lm:                     # scan input text up to char limit
            lc = c
            c = ts[i]                     # get next char
            if c in (COMMA, DOT):         # special treatment of COMMA and DOT
#               print 'comma'
                if ( not ellyChar.isDigit(lc) or
                     i + 1 == lm or
                     not ellyChar.isDigit(ts[i + 1])
                   ):
                    break
            elif c == DASH:
                if ( ellyChar.isDigit(lc) and
                     i + 1 < lm and
                     ellyChar.isDigit(ts[i + 1])
                   ):
                    break
            elif c == PERCENT:
                if len(sl) > 0:
                    break
            else:
                if not ellyChar.isLetterOrDigit(c):  # stop if not letter
                    if not c in ALSO: break          #   or "'" or "/" or "-"
            if c in (DOT, PERCENT, SLASH):
                if c in meeted:
                    break
                meeted.add(c)
            if c != COMMA:
                sl.append(c.lower())                 # otherwise append to sublist
            i += 1

#       print 'i=' , i , '<' + c + '>'

        if i < lm and ellyChar.isLetterOrDigit(ts[i]):     # proper termination?
            return 0                                       # if not, reject substring

        self.string = u''.join(sl)
        return i                          # scan count

    def rewriteNumber ( self , ts ):

        """
        check for number spelled out at start of text and rewrite in digit form if found

        arguments:
            self  -
            ts    - input text as a list of chars

        returns:
            True on any rewriting, False otherwise
        """

        vA = 0      # accumulated main value
        inited_from_num = False
        meet_float_mg = False
        dA = 0      # any denominator value
        lA = 0      # accumulated length of matched substring
        xA = ''     # ordinal ending
        sA = ''     # for ordinal plural

        salt = ''   # for alternative handling of hyphenated numbers

#       print 'rewrite: ts=' , ts
        t = ts
        tl = len(t)
        ns = 0      # space count

        while True: # collect all components of a number

            k = self.get(t)             # next component as string
#           print 'got k=' , k

            if k == 0: break            # if none, stop collecting

#           print 'gotten:' , self.string

            tl = self.string.split('-') # split up compound like 'twenty-four'

            lentl = len(tl)

#           print 'lentl=' , lentl

            if lentl > 3: break         # cannot have more than two hyphens

#           print 'tl=' , tl
            tl[-1] = tl[-1].rstrip('.')
            w = tl[0]                   # start of possible number
#           print 'w=' , w.encode('utf8')

            if w == '': break           # if current component starts with '-'

            if lentl == 1:              # no hyphenated part?
#               print 'no hyphen lA=' , lA , 'k=' , k
                if w in Card:           # if so, is it a cardinal
#                   print 'cardinal'
                    if type(vA) == float or inited_from_num:
                        break
                    vA += Card[w]       # if so, look up value and add to accumulation
                elif w in Ord:          # if not cardinal, check if ordinal
                    if type(vA) == float or inited_from_num:
                        break
                    if vA == 1:
                        dA = Ord[w]
                    elif vA == 0:
                        vA = Ord[w]
                    else:
                        left_num_str = str(vA)
                        right_num_str = str(Ord[w])
                        right_num_len = len(right_num_str)
                        if len(left_num_str) <= len(right_num_str) or '0' * right_num_len != left_num_str[-right_num_len:]:
                            dA = Ord[w]
                        else:
                            vA += Ord[w]    # if so, look up value and add to accumulation
                    xA = w[-2:]         # save 2 chars ordinal indicator, e.g. 'rd' of 'third'
                    lA += k
                    break               # ordinal always stops collection
                elif w in Magn:         # not cardinal or ordinal, is it a magnitude?
                    mg = Magn[w]        # if so, interpret magnitude
                    if type(mg) == float:
                        if meet_float_mg:
                            break
                        meet_float_mg = True

                    if vA == 0:         # no accumulated value value yet?
                        if type(mg) == float:
                            break
                        vA = mg         # if so, use magnitude as value
                    elif type(mg) != float and vA > mg:       # if accumulated value is greater than magnitude
                        md = vA%mg      # then extract right multiplier from accumulated value
                        vA -= md        #
                        if type(vA) == float or inited_from_num:
                            break
                        vA += md*mg     # add multiplied magnitude
                    else:
                        vA *= mg        # otherwise, just multiply by magnitude

                    if w[-2:] == 'th':
                        xA = 'th'       # if magnitude had ordinal form, mark it as such
                        lA += k
                        break           # ordinal stops collection
                elif vA == 0:
#                   print 'rewriteNumber w=' , w.encode('utf8')
                    try:
                        vA = int(w)     # expect integer value here
                        salt = w
                        inited_from_num = True
                    except ValueError:
                        try:                # otherwise, try float
                            vA = float(w)   # expect integer value here
                            salt = w
                            inited_from_num = True
                        except ValueError:
                            break           # otherwise, stop on failure to interpret
                elif w and w[-1] == 's' or w in ('half', 'halve'):
                    if vA != 1:         # interpret hyphenation as simple fraction
                        w = w[:-1]
                        sA = 's'        # for something like 'two-thirds'
#                       print 'x=' , x , 'sA=' , sA
                    if w == 'half' or w == 'halve':
                        w = 'second'    # normalize
                    elif not w in Ord or inited_from_num:
                        break
                    dA = Ord[w]
                    xA = w[-2:]         # mapping to simple fraction
                    lA += k
                    break
                else:
#                   print 'w=' , w , 'lA=' , lA
                    break
#               print 'continue scan lA=' , lA , 'k=' , k

            elif not w in Card:
                break                   # if non-cardinal, stop collection

            elif lentl == 3:            # expecting something like 'one-thirty-second'
#               print 'fraction tl[1]=' , tl[1]
                if tl[1] in Card:
                    dm10 = Card[tl[1]]
#                   print 'dm10=' , dm10
                    if dm10 > 10 and dm10 < 100:
                        dx = tl[2]
                        if dx[-1] == 's':
                            dx = dx[:-1]
                            sA = 's'
#                       print 'dx=' , dx , 'sA=' , sA
                        if dx in Ord and Ord[dx] < 10:
                            vA = Card[w]
                            dA = dm10 + Ord[dx]
                            xA = dx[-2:]
                            lA += k
#                           print 'vA=' , vA , 'dA=' , dA , 'xA=' , xA , 'lA=' , lA
                break

            elif lentl != 2:            # expecting expression like 'thirty-five'
                break                   #     or 'one-eighth'

            else:                       # otherwise, continue
                n = Card[w]             # get value of cardinal
#               print 'cardinal n=' , n
                x = tl[1]               # get second part of hyphenated number
                if len(x) == 0:
                    break

                if n == 1 or x[-1] == 's':
                    if n != 1:          # interpret hyphenation as simple fraction
                        if x not in Ord:
                            x = x[:-1]
                        sA = 's'        # for something like 'two-thirds'
#                       print 'x=' , x , 'sA=' , sA
                    if x == 'half' or x == 'halve':
                        x = 'second'    # normalize
                    elif x == 'quarter':
                        x = 'fourth'    # normalize
                    elif not x in Ord:
                        break
                    vA = n
                    dA = Ord[x]
                    xA = x[-2:]         # mapping to simple fraction
#                   print 'fraction=' , vA , dA , xA , sA

                else:
                    if ( not x in Ord ):    # must be written out number
                        break

                    if x in Card:           # get numerical values for two components
                        m = Card[x]         # cardinal
                    elif x in Ord:
                        m = Ord[x]          # ordinal
                        xA = x[-2:]         # and its marker

                    if n >= 20 and m < 10:
                        if type(vA) == float or inited_from_num:
                            break
                        vA += n + m         # make single number here if this makes sense
                        if xA != '':
                            lA += k
                            break           # ordinal stops collection
                    elif lA > 0:            # if cannot combine
                        break               # then stop scan if anything successfully parsed already
                    elif vA == 0:           # if nothing scanned, then recombine hyphenated parts
                        salt = str(n) + '/' + str(m)
                        lA += k
                        break               # have to stop scan
                    else:
                        break               # out of options, stop scan

#           print 'lA=' , lA , 'k=' , k
            lA += k                     # increment match char count
            t = t[k:]                   # look at next chars in input
            if t and t[0] == PERCENT:
                continue
            if len(t) < 2: break        # stop if not enough chars left

            if t[0] != ' ': break       # stop if next char is not space separator

            t = t[1:]                   # skip space
            ns += 1                     # and count it
#           print 'ns=' , ns

        lA += ns
#       print 'lA=' , lA
#       print 'ts=' , ts
        if lA == 0: return 0, ''  # if nothing matched, do nothing more

        if ts[lA-1] == ' ':             # must restore trailing space on match
            lA -= 1
        matched = lA

#       print 'ts=' , ts[:lA] , ts[lA:]

        while lA > 0:                   # remove matching substring
            ts.pop(0)
            lA -= 1

        if vA == 0:                     # no number found to rewrite?
            s = salt
        elif dA > 0:                    # a simple fraction found?
#           s = str(vA) + '/' + str(dA) + xA + sA
            s = str(vA) + '/' + str(dA)
        else:                           # numerical value as string + any ordinal indicator
            s = str(vA) + xA

#       print 's=' , s
        for c in s[::-1]:               # insert rewritten substring
            ts.insert(0,c)
        return matched, s

#
# unit test
#

if __name__ == '__main__':

    tdat = [    # test cases
        u'12 thousand' ,
        u'12345' ,
        u'12,345' ,
        u'1,000,000,000' ,
        u'forty-five' ,
        u'one hundred sixteen' ,
        u'three hundred thousand' ,
        u'three hundred thousand one hundred ninety-nine' ,
        u'ninety' ,
        u'twelve hundred four' ,
        u'forty-sixty' ,
        u'sixty-six hundred' ,
        u'thousand' ,
        u'million' ,
        u'one million million' ,
        u'twenty-eighth' ,
        u'hundred thirty-third' ,
        u'two hundred thirty-third' ,
        u'first national' ,
        u'two million two hundred thirty-five thousand eight hundred seventy-one' ,
        u'abe lincoln' ,
        u'four years' ,
        u'one.' ,
        u'one.hundred' ,
        u'one!' ,
        u'four.' ,
        u'four,' ,
        u'four yaks' ,
        u'twenty-one.' ,
        u'twenty-one,' ,
        u'twenty-one yaks' ,
        u'thirteenth' ,
        u'one-eighth' ,
        u'one-half' ,
        u'three-quarters' ,
        u'two-thirds' ,
        u'four-thirds' ,
        u'one-twentieth' ,
        u'101 %%' ,
        u'8%%' ,
        u'200 percent' ,
        u'200 percents' ,
        u'200 percents %' ,
        u'%%%' ,
        u'percent' ,
        u'1.1000. 100' ,
        u' 1.1000. 100' ,
        u'two-thirty' ,
        u'two-thirty-seconds'
    ]

    se = SimpleTransform()        # set up transformation object
    for xs in tdat:
        tst = list(xs)            # list of Unicode chars
        f = se.rewriteNumber(tst) # rewrite
        print(xs , '>>' , '+'.join(tst) , '=' , f)
