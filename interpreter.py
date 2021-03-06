# coding: utf-8
from sympy import binomial, factorial, Mul, Add, Float, Integer, Pow, sin, cos, tan, log, simplify
from sympy.parsing.sympy_parser import parse_expr


class Interpreter:

    unary_ops = {
        'tan': tan,
        'log': log,
        'cos': cos,
        'sin': sin,
        'factorial': factorial,
    }

    binary_ops = {
        'pow': Pow,
        'add': Add,
        'mul': Mul,
        'binomial': binomial,
    }

    def __init__(self, init_interp=None):
        if init_interp == None:
            self.stack = []
            self.op_count = 0
            self.valid_ops = self.plausible_ops(None)
        else:
            self.stack = init_interp.stack[:]
            self.op_count = init_interp.op_count
            self.valid_ops = set(init_interp.valid_ops)

    def load(self, x, stack=None):
        if stack is None:
            stack = self.stack
        if type(x) == int:
            x = Integer(x)
        elif type(x) == float:
            x = Integer(x) if x.is_integer() else Float(x)
        elif type(x) == str:
            x = parse_expr(x)
        stack.append(x)
        return x

    def binary_op(self, op, stack=None):
        if stack is None:
            stack = self.stack
        y = stack.pop()
        x = stack.pop()
        result = op(x, y, evaluate=False)
        assert result.is_real
        stack.append(result)
        return result

    def unary_op(self, op, stack=None):
        if stack is None:
            stack = self.stack
        x = stack.pop()
        result = op(x, evaluate=False)
        assert result.is_real
        stack.append(result)
        return result

    def plausible_ops(self, last_op):
        if last_op in ('end', 'exit'):
            return set()
        ops = {'load'}
        n = len(self.stack)
        if self.op_count == 0:
            ops.add('exit')
        if n == 1:
            ops.add('end')
        if n >= 1:
            for op_name, op_func in Interpreter.unary_ops.items():
                try:
                    self.unary_op(op_func, self.stack[-1:])
                    ops.add(op_name)
                except:
                    pass
        if n >= 2:
            for op_name, op_func in Interpreter.binary_ops.items():
                try:
                    self.binary_op(op_func, self.stack[-2:])
                    ops.add(op_name)
                except:
                    pass
        return ops

    def next_op(self, op_name, arg_num=None):
        assert op_name in self.valid_ops
        result = None
        if op_name == 'load':
            result = self.load(arg_num)
        elif op_name == 'end':
            assert len(self.stack) == 1
            result = simplify(self.stack.pop())
        elif op_name == 'exit':
            assert self.op_count == 0
        elif op_name in Interpreter.unary_ops:
            self.unary_op(Interpreter.unary_ops[op_name])
        elif op_name in Interpreter.binary_ops:
            self.binary_op(Interpreter.binary_ops[op_name])
        else:
            assert False
        self.op_count += 1
        self.valid_ops = self.plausible_ops(op_name)
        return len(self.valid_ops) == 0, result
