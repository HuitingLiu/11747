
# coding: utf-8

# In[1]:


import re


# In[2]:


from sympy import postorder_traversal, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, convert_xor, factorial_notation, function_exponentiation, implicit_multiplication, implicit_application, _token_splittable, split_symbols_custom
from sympy.parsing.sympy_tokenize import NUMBER, STRING, NAME, OP, ENDMARKER
from sympy.core.evaluate import global_evaluate
global_evaluate[0] = False


# In[3]:


from simpleTransform import SimpleTransform


# In[4]:


from collections import Counter


# In[5]:


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


def inspect(tokens, local_dict, global_dict):
    print(tokens)
    return tokens


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
    
    return (combinatorial_notation,) + standard_transformations + (
        convert_xor,
        split_symbols_custom(splittable),
        implicit_multiplication,
        implicit_application,
        function_exponentiation,
        #inspect
    )


# In[10]:


def all_symbols(expr):
    for sub_expr in postorder_traversal(expr):
        if hasattr(sub_expr, 'is_symbol') and sub_expr.is_symbol:
            yield str(sub_expr)


# In[11]:


def all_values(expr):
    for sub_expr in postorder_traversal(expr):
        eval_result = sub_expr.evalf()
        if eval_result.is_number:
            yield eval_result, len(sub_expr.args) == 0


# In[12]:


def parse(text, splittable_symbols=set(), local_dict={name: Symbol(name) for name in ('x', 'y', 'z', 'A', 'B', 'C')}):
    expr = parse_expr(text, local_dict=local_dict, transformations=get_transformations(splittable_symbols))
    for symbol in all_symbols(expr):
        if '_' in symbol and '' not in symbol.split('_'):
            continue
        if len(symbol) > 2:
            raise NameError()
        if len(symbol) == 2 and not symbol[-1].isdigit():
            raise NameError()
    return expr


# In[13]:


def try_parse(text, splittable_symbols=set()):
    try:
        return parse(text, splittable_symbols)
    except:
        return None


# In[14]:


def potential_expr(char):
    return char.isdigit() or char in {'+', '*', '/', '^'}


# In[15]:


def extract_exprs_from_line(line, splittable_symbols=set()):
    if not line:
        return []
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


def extract_exprs_from_text(text, splittable_symbols=set(), delimiter=re.compile(r'(=|\n|,|>|<|\(\w\))')):
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
            yield last_end, start, ''
        yield start, end, expr
        last_end = end
    if last_end < len(text):
        yield last_end, len(text), ''


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
        results.append((is_expr, segment, text[start:end]))

    return results

