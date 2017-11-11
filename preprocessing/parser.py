
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import re
import spacy

from collections import Counter
from simpleTransform import SimpleTransform
from sympy import postorder_traversal, Symbol
from sympy.parsing.sympy_parser import _token_splittable
from sympy.parsing.sympy_parser import convert_xor
from sympy.parsing.sympy_parser import factorial_notation
from sympy.parsing.sympy_parser import function_exponentiation
from sympy.parsing.sympy_parser import implicit_application
from sympy.parsing.sympy_parser import implicit_multiplication
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import split_symbols_custom
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_tokenize import ENDMARKER
from sympy.parsing.sympy_tokenize import NAME
from sympy.parsing.sympy_tokenize import NUMBER
from sympy.parsing.sympy_tokenize import OP
from sympy.parsing.sympy_tokenize import STRING


# In[3]:


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


# In[4]:


num_with_comma = re.compile(r'((\d+\.\d+|\d+|\?+|_+|\.+),)+\d+(\.\d+)?')

def process_num_with_comma(text):
    def islist(segment):
        if '?' in segment or '_' in segment:
            return True
        parts = segment.split(',')
        if any(map(lambda x: '.' in x, parts[:-1])):
            return True
        lengths = list(map(len, parts))
        lengths[-1] = len(parts[-1].split('.')[0])
        len_counts = Counter(lengths)
        if any(map(lambda x: x > 3, len_counts)):
            return True
        if len_counts[3] == 0:
            return True
        if len_counts[1] > 1 or len_counts[2] > 1:
            return True
        for left_length, right_length in zip(lengths, lengths[1:]):
            if left_length > right_length:
                return True
        if ',0' in segment:
            return False
        return len(parts) > 3
    result = ''
    rest_text = text
    while True:
        match = num_with_comma.search(rest_text)
        if match is None:
            result += rest_text
            break
        else:
            start, end = match.span()
            result += rest_text[:start]
            segment = match.group()
            if not islist(segment):
                result += segment
            else:
                result += ' , '.join(segment.split(','))
            rest_text = rest_text[end:]
    return result


# In[5]:


tokenizer = spacy.load('en').tokenizer
arrow_str = re.compile(r'(-( )*){3,}')
doller_num = re.compile(r'($)(\d)')
stock_num = re.compile(r'([A-z]\.)(\d)')
num_unit = re.compile(r'(\d,\d\d\d)([A-z])')
power_notation = re.compile(r'(\(\d+\))(\d)')
hyphen_concat = re.compile(r'(- )|( -)')

def rewrite_with_tokenization(text):
    text = power_notation.sub(r'\g<1>^\g<2>', text)
    text = arrow_str.sub('\u2192', text)
    text = doller_num.sub(r'\g<1> \g<2>', text)
    text = stock_num.sub(r'\g<1> \g<2>', text)
    text = num_unit.sub(r'\g<1> \g<2>', text)
    text = hyphen_concat.sub('-', text)
    text = process_num_with_comma(text)
    text = rewrite_number(text)
    spaced_text = ' '.join(map(str, tokenizer(text)))
    return rewrite_number(spaced_text)


# In[6]:


def combinatorial_notation(tokens, local_dict, global_dict):
    beginning = [(NAME, 'binomial'), (OP, '(')]
    comma = [(OP, ',')]
    end = [(OP, ')')]
    last_toknum = None
    result = []
    for toknum, tokval in tokens:
        if last_toknum == NUMBER and toknum == NAME and len(tokval) > 1 and tokval[0] in ('c', 'C') and tokval[1:].isdigit():
            result = result[:-1] + beginning + result[-1:] + comma + [(NUMBER, tokval[1:])] + end
        else:
            result.append((toknum, tokval))
            last_toknum = toknum
    return result


# In[7]:


operator_dict = {
    '\u00f7': '/',
    '\u00d7': '*',
}
def unicode_operator(tokens, local_dict, global_dict):
    result = []
    for toknum, tokval in tokens:
        if tokval in operator_dict:
            toknum = OP
            tokval = operator_dict[tokval]
        result.append((toknum, tokval))
    return result


# In[8]:


def reject_symbols(symbols):
    def transformation(tokens, local_dict, global_dict):
        for toknum, tokval in tokens:
            if toknum == NAME and tokval in symbols:
                raise NameError()
        return tokens
    return transformation


# In[9]:


def get_transformations(splittable_symbols):
    def splittable(symbol):
        return set(symbol).issubset(splittable_symbols)
    return (combinatorial_notation, unicode_operator) + standard_transformations + (
        convert_xor,
        split_symbols_custom(splittable),
        implicit_multiplication,
        implicit_application,
        function_exponentiation
    )


# In[10]:


def all_symbols(expr):
    for sub_expr in postorder_traversal(expr):
        if hasattr(sub_expr, 'is_symbol') and sub_expr.is_symbol:
            yield str(sub_expr)


# In[11]:


def all_values(expr):
    args = expr.args
    if len(args) >= 1:
        yield from all_values(args[0])
    if len(args) >= 2:
        yield from all_values(args[1])
    if len(args) >= 3:
        partial_expr = expr.func(args[0], args[1])
        for arg in args[2:]:
            partial_eval_result = partial_expr.evalf()
            if partial_eval_result.is_number:
                yield partial_eval_result, partial_expr
            yield from all_values(arg)
            partial_expr = expr.func(partial_expr, arg)
    eval_result = expr.evalf()
    if eval_result.is_number:
        yield eval_result, expr


# In[12]:


def parse(text, splittable_symbols=set(), local_dict={name: Symbol(name) for name in ('x', 'y', 'z', 'A', 'B', 'C', 'I')}):
    from sympy import binomial, factorial, factorial2, Mul, Add
    local_dict['binomial'] = lambda x, y: binomial(x, y, evaluate = False)
    local_dict['factorial'] = lambda x: factorial(x, evaluate = False)
    local_dict['factorial2'] = lambda x: factorial2(x, evaluate = False)
    try:
        mul_identity = Mul.identity
        Mul.identity = None
        add_identity = Add.identity
        Add.identity = None
        expr = parse_expr(text, local_dict=local_dict, transformations=get_transformations(splittable_symbols), evaluate=False)
    finally:
        Mul.identity = mul_identity
        Add.identity = add_identity
    for symbol in all_symbols(expr):
        if '_' in symbol and '' not in symbol.split('_'):
            continue
        if len(symbol) > 2:
            raise NameError()
        if len(symbol) == 2 and not symbol[-1].isdigit():
            raise NameError()
    return expr


# In[13]:


def try_parse(text, splittable_symbols=set(), local_dict=None):
    try:
        if local_dict is None:
            return parse(text, splittable_symbols=splittable_symbols)
        return parse(text, splittable_symbols=splittable_symbols, local_dict=local_dict)
    except:
        return None


# In[14]:


def potential_expr(char):
    return char.isdigit() or char in {'+', '*', '/', '^'}


# In[15]:


def extract_exprs_from_line(line, splittable_symbols=set()):
    if not line:
        return
    pos = 0
    n = len(line)
    lower_bound = 0
    while pos < n:
        if potential_expr(line[pos]):
            start = pos
            end = pos + 1
            found = False
            for start in range(lower_bound, pos + 1):
                if start > lower_bound and line[start-1].isalpha() and line[start].isalpha():
                    continue
                for end in range(n, pos, -1):
                    if end < n and line[end].isalpha() and line[end-1].isalpha():
                        continue
                    expr_text = line[start:end]
                    expr = try_parse(expr_text, splittable_symbols)
                    if expr is not None:
                        yield start, end, expr
                        lower_bound = end
                        pos = end
                        found = True
                        break
                if found:
                    break
            if not found:
                pos += 1
        else:
            pos += 1


# In[16]:


def extract_exprs_from_text(text, splittable_symbols=set(), delimiter=re.compile(r'(=|\n|,|>|<)')):
    base = 0
    for segment in delimiter.split(text):
        if len(segment) > 0 and delimiter.match(segment) is None:
            for start, end, expr in extract_exprs_from_line(segment, splittable_symbols):
                yield base + start, base + end, expr
        base += len(segment)


# In[17]:


def split_text_and_expr(text, splittable_symbols=set()):
    last_end = 0
    for start, end, expr in extract_exprs_from_text(text, splittable_symbols):
        if last_end != start:
            yield last_end, start, text[last_end:start]
        yield start, end, expr
        last_end = end
    if last_end < len(text):
        yield last_end, len(text), text[last_end:len(text)]


# In[18]:


def parse_rationale(text):
    symbols = set()
    for start, end, segment in split_text_and_expr(text):
        is_expr = type(segment) != str
        if is_expr:
            symbols |= set(all_symbols(segment))
    splittable_symbols = {symbol for symbol in symbols if len(symbol) == 1}
    #print(splittable_symbols)

    results = []
    for start, end, segment in split_text_and_expr(text, splittable_symbols=splittable_symbols):
        is_expr = type(segment) != str
        if is_expr:
            results.append((is_expr, segment))
        else:
            results.append((is_expr, list(map(lambda x:str(x).lower(), tokenizer(segment)))))

    return results


# In[19]:


def extract_nums(text):
    result = []
    for is_expr, expr in parse_rationale(text):
        if is_expr:
            nums = list(all_values(expr))
            for index, (num, sub_expr) in enumerate(nums):
                result.append((num, len(sub_expr.args) == 0, index + 1 == len(nums)))
    return result

