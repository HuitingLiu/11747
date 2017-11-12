
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import copy
import re
import spacy

from collections import Counter
from simpleTransform import SimpleTransform
from sympy import postorder_traversal, simplify
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
hyphen_concat = re.compile(r' ?- ?')

def rewrite_with_tokenization(text):
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"')
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
    '\u2013': [(OP, '-')],
    '\u2014': [(OP, '-')],
    '\u2212': [(OP, '-')],
    '\u00f7': [(OP, '/')],
    '\u2044': [(OP, '/')],
    '\u2217': [(OP, '*')],
    '\u00d7': [(OP, '*')],
    '\u02c6': [(OP, '^')],
    '\u00b2': [(OP, '^'), (NUMBER, '2')],
    '\u00b3': [(OP, '^'), (NUMBER, '3')],
    '\u00b9': [(OP, '^'), (NUMBER, '1')],
    '\u2070': [(OP, '^'), (NUMBER, '0')],
    '\u2074': [(OP, '^'), (NUMBER, '4')],
    '\u2075': [(OP, '^'), (NUMBER, '5')],
    '\u2076': [(OP, '^'), (NUMBER, '6')],
    '\u2077': [(OP, '^'), (NUMBER, '7')],
    '\u2078': [(OP, '^'), (NUMBER, '8')],
    '\u2079': [(OP, '^'), (NUMBER, '9')],
    '\u221a': [(NAME, 'sqrt')],
    '\u2154': [(OP, '('), (NUMBER, '2'), (OP, '/'), (NUMBER, '3'), (OP, ')')],
    '\u00bd': [(OP, '('), (NUMBER, '1'), (OP, '/'), (NUMBER, '2'), (OP, ')')],
    '\u00bc': [(OP, '('), (NUMBER, '1'), (OP, '/'), (NUMBER, '4'), (OP, ')')],
    '\u00be': [(OP, '('), (NUMBER, '3'), (OP, '/'), (NUMBER, '4'), (OP, ')')],
    '\u2157': [(OP, '('), (NUMBER, '3'), (OP, '/'), (NUMBER, '5'), (OP, ')')],
    '\u2158': [(OP, '('), (NUMBER, '4'), (OP, '/'), (NUMBER, '5'), (OP, ')')],
    '\u03c0': [(NAME, 'pi')],
    '[': [(OP, '(')],
    ']': [(OP, ')')],
    '{': [(OP, '(')],
    '}': [(OP, ')')],
}
operator_splitor = re.compile('(%s)' % '|'.join(operator_dict.keys()))
def unicode_operator(tokens, local_dict, global_dict):
    result = []
    for toknum, tokval in tokens:
        if tokval in operator_dict:
            result += operator_dict[tokval]
        else:
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


def contains_symbol(expr):
    exprs_with_symbol = set()
    def contains(expr):
        expr_id = id(expr)
        return expr_id in exprs_with_symbol
    for sub_expr in enumerate(postorder_traversal(expr)):
        if hasattr(sub_expr, 'is_symbol') and sub_expr.is_symbol or (hasattr(sub_expr, 'args') and any(map(lambda x:contains(x), sub_expr.args))):
            exprs_with_symbol.add(id(sub_expr))
    return contains


# In[12]:


def all_values(expr):
    
    has_symbol = contains_symbol(expr)
    
    def traversal(expr):
        args = expr.args
        if len(args) >= 1:
            yield from traversal(args[0])
        if len(args) >= 2:
            yield from traversal(args[1])
        if len(args) >= 3:
            partial_expr = expr.func(args[0], args[1])
            for arg in args[2:]:
                if has_symbol(partial_expr):
                    break
                partial_eval_result = partial_expr.evalf()
                if partial_eval_result.is_number:
                    yield partial_eval_result, partial_expr
                yield from traversal(arg)
                partial_expr = expr.func(partial_expr, arg)
        if has_symbol(expr):
            return
        eval_result = expr.evalf()
        if eval_result.is_number:
            yield eval_result, expr
    
    yield from traversal(expr)


# In[13]:


def genereate_local_dict():
    from sympy import binomial, factorial, factorial2, Mul, Add, Integer, Pow, Float, Symbol, sin, cos, tan, log, sqrt, pi
    local_dict = {name: Symbol(name) for name in ('x', 'y', 'z', 'A', 'B', 'C', 'I', 'S', 'i', 's')}
    local_dict['binomial'] = lambda x, y: binomial(x, y, evaluate = False)
    local_dict['factorial'] = lambda x: factorial(x, evaluate = False)
    local_dict['factorial2'] = lambda x: factorial2(x, evaluate = False)
    local_dict['Mul'] = copy.deepcopy(Mul)
    local_dict['Mul'].identity = copy.deepcopy(Mul.identity)
    local_dict['Add'] = copy.deepcopy(Add)
    local_dict['Add'].identity = copy.deepcopy(Add.identity)
    local_dict['Integer'] = Integer
    local_dict['Pow'] = Pow
    local_dict['Float'] = Float
    local_dict['Symbol'] = Symbol
    local_dict['sin'] = sin
    local_dict['cos'] = cos
    local_dict['tan'] = tan
    local_dict['log'] = log
    local_dict['sqrt'] = sqrt
    local_dict['pi'] = pi
    return local_dict


# In[14]:


def parse(text, splittable_symbols=set(), local_dict=genereate_local_dict()):
    from sympy.core.assumptions import ManagedProperties
    from sympy.core.function import FunctionClass, UndefinedFunction
    from sympy import Tuple
    from types import FunctionType, MethodType
    from sympy.logic.boolalg import BooleanTrue, BooleanFalse
    text = ' '.join(operator_splitor.split(text))
    expr = parse_expr(text, local_dict=local_dict, global_dict={}, transformations=get_transformations(splittable_symbols), evaluate=False)
    if any(map(lambda x: type(x) in (BooleanTrue, BooleanFalse, ManagedProperties, FunctionClass, FunctionType, MethodType, Tuple, tuple, float, int, str, bool) or type(type(x)) in (UndefinedFunction, ), postorder_traversal(expr))):
        raise NameError()
    for symbol in all_symbols(expr):
        if '_' in symbol and '' not in symbol.split('_'):
            continue
        if len(symbol) > 2:
            raise NameError()
        if len(symbol) == 2 and not symbol[-1].isdigit():
            raise NameError()
    return expr


# In[15]:


def try_parse(text, splittable_symbols=set(), local_dict=None):
    try:
        if local_dict is None:
            return parse(text, splittable_symbols=splittable_symbols)
        return parse(text, splittable_symbols=splittable_symbols, local_dict=local_dict)
    except:
        return None


# In[16]:


token_pattern = re.compile(r'[^\W\d_]+|\d+|\W|_+')
def extract_exprs_from_line(line, splittable_symbols=set()):
    if not line:
        return
    pos_list = [0]
    potential_expr = []
    for token in token_pattern.findall(line):
        if token.isspace():
            pos_list[-1] += len(token)
        else:
            potential_expr.append(token.isdigit() or token in {'+', '*', '/', '^'} or token in operator_dict)
            pos_list.append(pos_list[-1] + len(token))
    n = len(pos_list)
    potential_expr.append(False)
    pos_index = 0
    lower_bound = 0
    while pos_index < n:
        found = False
        if potential_expr[pos_index]:
            for start_index in range(lower_bound, pos_index+1):
                for end_index in range(n - 1, pos_index, -1):
                    start, end = pos_list[start_index], pos_list[end_index]
                    expr_text = line[start:end]
                    expr = try_parse(expr_text, splittable_symbols)
                    if expr is not None:
                        yield start, end, expr
                        lower_bound = end_index
                        pos_index = end_index
                        found = True
                        break
                if found:
                    break
        if not found:
            pos_index += 1


# In[17]:


def extract_exprs_from_text(text, splittable_symbols=set(), delimiter=re.compile(r'(=|\n|,|>|<|~|\$)')):
    base = 0
    for segment in delimiter.split(text):
        if len(segment) > 0 and delimiter.match(segment) is None:
            for start, end, expr in extract_exprs_from_line(segment, splittable_symbols):
                yield base + start, base + end, expr
        base += len(segment)


# In[18]:


def split_text_and_expr(text, splittable_symbols=set()):
    last_end = 0
    for start, end, expr in extract_exprs_from_text(text, splittable_symbols):
        if last_end != start:
            yield last_end, start, text[last_end:start]
        yield start, end, expr
        last_end = end
    if last_end < len(text):
        yield last_end, len(text), text[last_end:len(text)]


# In[19]:


def parse_rationale(text):
    text = rewrite_with_tokenization(text)
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


# In[20]:


def extract_nums(text):
    result = []
    for is_expr, expr in parse_rationale(text):
        if is_expr:
            nums = list(all_values(expr))
            for index, (num, sub_expr) in enumerate(nums):
                result.append((num, len(sub_expr.args) == 0, index + 1 == len(nums)))
    return result

