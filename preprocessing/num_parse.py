from __future__ import print_function
from simpleTransform import SimpleTransform

def rewrite_number(text):
    se = SimpleTransform()
    start = 0
    n = len(text)
    result = ''
    while start < n:
        matched, num_str = se.rewriteNumber(list(text[start:]))
        if matched > 0:
            start += matched
            result += num_str
        else:
            while start < n and not text[start].isspace():
                result += text[start]
                start += 1
            if start < n:
                result += text[start]
            start += 1
    return result
