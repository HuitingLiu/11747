# coding: utf-8

import dynet as dy
import numpy as np
import argparse
import pickle
import itertools
import random
import math
import os
import heapq

from contextlib import contextmanager
from collections import Iterable, defaultdict
from itertools import chain
from sympy.parsing.sympy_parser import parse_expr
from interpreter import Interpreter
from preprocessing.parser import parse_question, extract_instructions
from timeout_decorator import timeout
from decimal import Decimal

UNK = ('<UNK>',)


@contextmanager
def parameters(*params):
    result = tuple(map(lambda x: dy.parameter(x), params))
    if len(result) == 0:
        yield None
    elif len(result) == 1:
        yield result[0]
    else:
        yield result


def decompose(x):
    num = Decimal(x)
    sign, digits, exponent = num.as_tuple()
    fexp = float(len(digits) + exponent - 1)
    fman = float(num.scaleb(-fexp).normalize())
    return fexp, fman


def val_embed(val):
    isfloat = False
    isnan = False
    isinf = False
    isneg = False
    fexp = 0.0
    fman = 0.0
    try:
        val = float(val)
        isfloat = True
        isnan = math.isnan(val)
        isinf = math.isinf(val)
        isneg = val < 0
        fexp, fman = decompose(val)
    except:
        pass
    return dy.inputVector([float(isfloat), float(isnan), float(isinf), float(isneg), fexp, fman])


class Embedder(object):
    def __init__(self, model, obj2id, embed_dim):
        assert min(obj2id.values()) >= 0, 'Cannot embed negative id'

        model = self.model = model.add_subcollection(self.__class__.__name__)
        self.spec = obj2id, embed_dim

        vocab_size = max(obj2id.values()) + 1
        self.embeds = model.add_lookup_parameters((vocab_size, embed_dim))
        self.obj2id = obj2id

    @classmethod
    def from_spec(cls, spec, model):
        obj2id, embed_dim = spec
        return cls(model, obj2id, embed_dim)

    def param_collection(self):
        return self.model

    def __getitem__(self, key):
        i = key
        if type(key) != int:
            i = self.obj2id[key if key in self.obj2id else UNK]
        return self.embeds[i]

    def __call__(self, seq):
        return [self[x] for x in seq]


class Linear(object):
    def __init__(self, model, input_dim, output_dim):
        model = self.model = model.add_subcollection(self.__class__.__name__)
        self.spec = input_dim, output_dim

        self.W = model.add_parameters((output_dim, input_dim))
        self.b = model.add_parameters(output_dim)

    @classmethod
    def from_spec(cls, spec, model):
        input_dim, output_dim = spec
        return cls(model, input_dim, output_dim)

    def param_collection(self):
        return self.model

    def __call__(self, input_expr):
        with parameters(self.W, self.b) as (W, b):
            return dy.affine_transform([b, W, input_expr])


class Encoder(object):
    def __init__(self, model, word2wid, word_embed_dim, num_layers, hidden_dim):
        assert hidden_dim % 2 == 0, "BiLSTM hidden dimension must be even."

        self.word2wid = word2wid
        self.words_embeds = Embedder(model, word2wid, word_embed_dim)

        model = self.model = model.add_subcollection(self.__class__.__name__)
        self.spec = word2wid, word_embed_dim, num_layers, hidden_dim

        val_repr_dim = 6
        self.fwdLSTM = dy.LSTMBuilder(num_layers, word_embed_dim + val_repr_dim, hidden_dim / 2, model)
        self.bwdLSTM = dy.LSTMBuilder(num_layers, word_embed_dim + val_repr_dim, hidden_dim / 2, model)

    @classmethod
    def from_spec(cls, spec, model):
        word2wid, word_embed_dim, num_layers, hidden_dim = spec
        return cls(model, word2wid, word_embed_dim, num_layers, hidden_dim)

    def param_collection(self):
        return self.model

    def set_dropout(self, p):
        self.fwdLSTM.set_dropout(p)
        self.bwdLSTM.set_dropout(p)

    def disable_dropout(self):
        self.fwdLSTM.disable_dropout()
        self.bwdLSTM.disable_dropout()

    def embed_with_val(self, words):
        result = []
        for word, val in words:
            word_embed = self.words_embeds[word]
            val_repr = val_embed(val)
            result.append(dy.concatenate([word_embed, val_repr]))
        return result

    def __call__(self, question, options):
        q_seq = self.embed_with_val(question)

        fqs = self.fwdLSTM.initial_state().add_inputs(q_seq)
        bqs = self.bwdLSTM.initial_state().add_inputs(reversed(q_seq))
        es = [dy.concatenate([f.output(), b.output()]) for f, b in zip(fqs, reversed(bqs))]

        s = [dy.concatenate([x, y]) for x, y in zip(fqs[-1].s(), bqs[-1].s())]

        option_embeds = []
        for option in options:
            o_seq = self.embed_with_val(option)
            fos = fqs[-1].transduce(o_seq)
            bos = bqs[-1].transduce(reversed(o_seq))
            es.extend(dy.concatenate([f, b]) for f, b in zip(fos, reversed(bos)))
            option_embeds.append(es[-1])

        return es, s, option_embeds


class Attender(object):
    def __init__(self, model, query_dim, content_dim, att_dim):
        model = self.model = model.add_subcollection(self.__class__.__name__)
        self.spec = query_dim, content_dim, att_dim

        self.P = model.add_parameters((att_dim, content_dim))
        self.W = model.add_parameters((att_dim, query_dim))
        self.b = model.add_parameters((1, att_dim))

    @classmethod
    def from_spec(cls, spec, model):
        query_dim, content_dim, att_dim = spec
        return cls(model, query_dim, content_dim, att_dim)

    def param_collection(self):
        return self.model

    def __call__(self, es=None):
        with parameters(self.P, self.W, self.b) as (P, W, b):
            if es:
                assert len(es) > 0

                class ImmutableState(object):
                    def __init__(self):
                        self.es_matrix = dy.concatenate_cols(es)
                        self.ps_matrix = P * self.es_matrix

                    def cal_scores(self, s):
                        hs_matrix = dy.tanh(dy.colwise_add(self.ps_matrix, W * s))
                        return dy.softmax(dy.transpose(b * hs_matrix))

                    def cal_context(self, s, selected=None):
                        ws = self.cal_scores(s)
                        if selected is None:
                            return self.es_matrix * ws, ws
                        selected_ws = dy.select_rows(ws, selected)
                        selected_ws = dy.cdiv(selected_ws, dy.sum_elems(selected_ws))
                        return dy.concatenate_cols([es[index] for index in selected]) * selected_ws, ws

                return ImmutableState()

            else:

                class MutableSate(object):
                    def __init__(self, es, ps):
                        self.es = es
                        self.ps = ps

                    def append_e(self, e):
                        return MutableSate(self.es + [e], self.ps + [P * e])

                    def cal_scores(self, s):
                        if len(self.ps) == 0:
                            return None
                        hs_matrix = dy.tanh(dy.colwise_add(dy.concatenate_cols(self.ps), W * s))
                        return dy.softmax(dy.transpose(b * hs_matrix))

                    def cal_context(self, s, selected=None):
                        ws = self.cal_scores(s)
                        if ws is None:
                            return None, None
                        if selected is None:
                            return dy.concatenate_cols(self.es) * ws, ws
                        selected_ws = dy.select_rows(ws, selected)
                        selected_ws = dy.cdiv(selected_ws, dy.sum_elems(selected_ws))
                        return dy.concatenate_cols([self.es[index] for index in selected]) * selected_ws, ws

                return MutableSate([], [])


class Decoder(object):
    def __init__(self, model, op_names, op_embed_dim, prior_nums, num_embed_dim, sign_embed_dim, encode_dim, att_dim,
                 num_layers, hidden_dim):
        assert encode_dim == hidden_dim, "Encoder dimension dosen't match with decoder dimension"
        assert encode_dim == num_embed_dim, "Number embedding dimension doesn't match with encoder dimension"
        assert type(op_names) == list

        self.spec = op_names, op_embed_dim, prior_nums, num_embed_dim, sign_embed_dim, encode_dim, att_dim, \
                    num_layers, hidden_dim
        model = self.model = model.add_subcollection(self.__class__.__name__)

        num_ops = len(op_names)
        self.name2opid = {name: opid for opid, name in enumerate(op_names)}
        self.opid2name = {opid: name for opid, name in enumerate(op_names)}
        self.op_embeds = Embedder(model, self.name2opid, op_embed_dim)
        self.start_op = model.add_parameters(op_embed_dim)

        self.prior_nums = prior_nums
        self.nid2num = {nid: num for num, nid in prior_nums.items()}
        self.num_embeds = Embedder(model, prior_nums, num_embed_dim)
        self.dummy_arg = model.add_parameters(num_embed_dim + sign_embed_dim)
        self.arg_dim = num_embed_dim + sign_embed_dim

        self.neg_embed = model.add_parameters(sign_embed_dim)
        self.pos_embed = model.add_parameters(sign_embed_dim)

        self.neg_input_embed = model.add_parameters(sign_embed_dim)
        self.pos_input_embed = model.add_parameters(sign_embed_dim)
        self.neg_exprs_embed = model.add_parameters(sign_embed_dim)
        self.pos_exprs_embed = model.add_parameters(sign_embed_dim)
        self.neg_prior_embed = model.add_parameters(sign_embed_dim)
        self.pos_prior_embed = model.add_parameters(sign_embed_dim)

        self.decode_att = Attender(model, hidden_dim, encode_dim, att_dim)
        self.input_att = Attender(model, hidden_dim + sign_embed_dim, encode_dim, att_dim)
        self.exprs_att = Attender(model, hidden_dim + sign_embed_dim, hidden_dim, att_dim)
        self.option_att = Attender(model, hidden_dim, encode_dim, att_dim)

        num_copy_src = 6
        self.h2op = Linear(model, hidden_dim, num_ops)
        self.h2ht = Linear(model, hidden_dim, hidden_dim)
        self.h2copy = Linear(model, hidden_dim, num_copy_src)

        num_prior = len(prior_nums)
        self.sh2prior = Linear(model, hidden_dim + sign_embed_dim, num_prior)

        expr_val_dim = 6
        self.opLSTM = dy.LSTMBuilder(num_layers, encode_dim + expr_val_dim + op_embed_dim + self.arg_dim, hidden_dim, model)

    @classmethod
    def from_spec(cls, spec, model):
        op_names, op_embed_dim, prior_nums, num_embed_dim, sign_embed_dim, encode_dim, att_dim, num_layers, \
        hidden_dim = spec
        return cls(model, op_names, op_embed_dim, prior_nums, num_embed_dim, sign_embed_dim, encode_dim, att_dim,
                   num_layers, hidden_dim)

    def param_collection(self):
        return self.model

    def set_dropout(self, p):
        self.opLSTM.set_dropout(p)

    def disable_dropout(self):
        self.opLSTM.disable_dropout()

    def __call__(self, es, e, option_embeds, input_num_indexes):
        assert isinstance(input_num_indexes, Iterable)
        decoder = self
        context_atts = decoder.decode_att(es)
        input_num_indexes = sorted(set(input_num_indexes))
        input_atts = decoder.input_att([es[index] for index in input_num_indexes])
        option_atts = decoder.option_att(option_embeds)

        class State(object):
            def __init__(self, s, step, exprs_num_indexs, expr_atts):
                self.s = s
                self.step = step
                self.exprs_num_indexs = exprs_num_indexs
                self.expr_atts = expr_atts

            def op_probs(self):
                h = self.s.output()
                return dy.softmax(decoder.h2op(h))

            def op_prob(self, op):
                probs = self.op_probs()
                if type(op) == str:
                    op = decoder.name2opid[op]
                return dy.pick(probs, op)

            def copy_probs(self):
                h = self.s.output()
                return dy.softmax(decoder.h2copy(h))

            def _signed_h(self, h=None, neg=False):
                with parameters(decoder.neg_embed, decoder.pos_embed) as (neg_embed, pos_embed):
                    signed_h = dy.concatenate([h if h is not None else self.s.output(), neg_embed if neg else pos_embed])
                return signed_h

            def from_prior_probs(self, neg=False):
                signed_h = self._signed_h(neg=neg)
                return dy.softmax(decoder.sh2prior(signed_h))

            def from_prior_prob(self, num, neg=False):
                probs = self.from_prior_probs(neg)
                if type(num) == float:
                    num = decoder.prior_nums[num if num in decoder.prior_nums else UNK]
                if decoder.nid2num[num] == UNK:
                    return dy.scalarInput(0.0), dy.zeros(decoder.arg_dim)
                with parameters(decoder.neg_prior_embed, decoder.pos_prior_embed) as (neg_prior_embed, pos_prior_embed):
                    prior_ref = dy.concatenate([decoder.num_embeds[num], neg_prior_embed if neg else pos_prior_embed])
                return dy.pick(probs, num), prior_ref

            def from_input_probs(self, neg=False):
                signed_h = self._signed_h(neg=neg)
                return input_atts.cal_scores(signed_h)

            def from_input_prob(self, selected_indexes, neg=False):
                assert type(selected_indexes) == set
                selected_indexes = [index for index, old_index in enumerate(input_num_indexes) if old_index in selected_indexes]
                if len(selected_indexes) == 0:
                    return dy.scalarInput(0.0), dy.zeros(decoder.arg_dim)
                signed_h = self._signed_h(neg=neg)
                input_ref, probs = input_atts.cal_context(signed_h, selected_indexes)
                with parameters(decoder.neg_input_embed, decoder.pos_input_embed) as (neg_input_embed, pos_input_embed):
                    input_ref = dy.concatenate([input_ref, neg_input_embed if neg else pos_input_embed])
                return dy.sum_elems(dy.select_rows(probs, selected_indexes)), input_ref

            def from_exprs_probs(self, neg=False):
                ht = dy.tanh(decoder.h2ht(self.s.output()))
                signed_h = self._signed_h(ht, neg)
                return self.expr_atts.cal_scores(signed_h)

            def from_exprs_prob(self, selected_indexes, neg=False):
                assert type(selected_indexes) == set
                selected_indexes = [index for index, old_index in enumerate(self.exprs_num_indexs) if old_index in selected_indexes]
                if len(selected_indexes) == 0:
                    return dy.scalarInput(0.0), dy.zeros(decoder.arg_dim)
                ht = dy.tanh(decoder.h2ht(self.s.output()))
                signed_h = self._signed_h(ht, neg)
                exprs_ref, probs = self.expr_atts.cal_context(signed_h, selected_indexes)
                with parameters(decoder.neg_exprs_embed, decoder.pos_exprs_embed) as (neg_exprs_embed, pos_exprs_embed):
                    exprs_ref = dy.concatenate([exprs_ref, neg_exprs_embed if neg else pos_exprs_embed])
                return dy.sum_elems(dy.select_rows(probs, selected_indexes)), exprs_ref

            def next_state(self, expr_val, op, arg_ref=None):
                op_embed = decoder.op_embeds[op] if type(op) in (int, str) else op
                with parameters(decoder.dummy_arg) as dummy_arg:
                    arg_embed = dummy_arg if arg_ref is None else arg_ref
                context, _ = context_atts.cal_context(self.s.output())
                next_s = self.s.add_input(dy.concatenate([context, val_embed(expr_val), op_embed, arg_embed]))
                if op in ('end', decoder.name2opid['end']):
                    return State(next_s, self.step + 1, self.exprs_num_indexs + [self.step], self.expr_atts.append_e(next_s.output()))
                return State(next_s, self.step + 1, self.exprs_num_indexs, self.expr_atts)

            def predict_answer(self):
                h = self.s.output()
                return option_atts.cal_scores(h)

        with parameters(decoder.start_op) as start_op:
            return State(decoder.opLSTM.initial_state().set_s(e), -1, [], decoder.exprs_att()).next_state(0.0, start_op)


def add_expr_val(dataset):
    @timeout(10, use_signals=False)
    def process_trace(trace):
        new_trace = []
        interp = Interpreter()
        for instruction in trace:
            instruction = tuple(instruction)
            op_name = instruction[0]
            arg_num = None if len(instruction) < 2 else instruction[-1]
            end_expr, expr_val = interp.next_op(op_name, arg_num)
            if end_expr:
                interp = Interpreter()
            if expr_val is not None:
                expr_val = float(expr_val)
            new_trace.append(instruction[:1] + (expr_val,)+ instruction[1:])
        return new_trace

    result = []
    for question, options, trace, input_num_indexes, answer in dataset:
        try:
            result.append((question, options, process_trace(trace), input_num_indexes, answer))
        except:
            pass
    return result


def process_words(words):
    words_with_val = []
    for word in words:
        val = None
        try:
            expr = parse_expr(word)
            if expr.is_number:
                val = float(expr)
        except:
            pass
        words_with_val.append((word, val))
    return words_with_val


def add_num_val(dataset):
    result = []
    for question, options, trace, input_num_indexes, answer in dataset:
        question = process_words(question)
        options = [process_words(option) for option in options]
        result.append((question, options, trace, input_num_indexes, answer))
    return result



def build_index(question, options):
    input_nums = {}
    for index, token in enumerate(chain(*([question] + options))):
        try:
            val = parse_expr(token)
            if val.is_number:
                input_nums[index] = val
        except:
            pass
    return input_nums


def solve(encoder, decoder, raw_question, raw_options, max_op_count):
    copy_id2src = ['pos_prior', 'neg_prior', 'pos_input', 'neg_input', 'pos_exprs', 'neg_exprs']
    question = parse_question(raw_question)
    options = [parse_question(raw_option) for raw_option in raw_options]
    input_nums = build_index(question, options)
    input_nums_indexes = sorted(input_nums.keys())
    UNK_id = decoder.prior_nums[UNK]
    dy.renew_cg()
    es, e, option_embeds = encoder(process_words(question), map(process_words, options))
    state = decoder(es, e, option_embeds, input_nums.keys())
    interp = Interpreter()
    expr_vals = []
    ds = []
    trace = []
    for _ in range(max_op_count):
        p_op = state.op_probs().npvalue()
        p_op[[op_name not in interp.valid_ops for op_id, op_name in decoder.opid2name.items()]] = -np.inf
        op_name = decoder.opid2name[p_op.argmax()]
        arg_num = None
        arg_ref = None
        if op_name == 'load':
            copy_src = copy_id2src[state.copy_probs().npvalue().argmax()]
            neg = copy_src.startswith('neg')
            if copy_src.endswith('_input'):
                num_index = input_nums_indexes[state.from_input_probs(neg=neg).npvalue().argmax()]
                arg_num = input_nums[num_index]
                with parameters(decoder.neg_input_embed, decoder.pos_input_embed) as (neg_input_embed, pos_input_embed):
                    arg_ref = dy.concatenate([es[num_index], neg_input_embed if neg else pos_input_embed])
            elif copy_src.endswith('_prior'):
                p_prior = state.from_prior_probs(neg=neg).npvalue()
                p_prior[UNK_id] = -np.inf
                nid = p_prior.argmax()
                arg_num = decoder.nid2num[nid]
                with parameters(decoder.neg_prior_embed, decoder.pos_prior_embed) as (neg_prior_embed, pos_prior_embed):
                    arg_ref = dy.concatenate([decoder.num_embeds[nid], neg_prior_embed if neg else pos_prior_embed])
            elif copy_src.endswith('_exprs'):
                expr_index = state.from_exprs_probs(neg=neg).npvalue().argmax()
                arg_num = expr_vals[expr_index]
                with parameters(decoder.neg_exprs_embed, decoder.pos_exprs_embed) as (neg_exprs_embed, pos_exprs_embed):
                    arg_ref = dy.concatenate([ds[expr_index], neg_exprs_embed if neg else pos_exprs_embed])
            else:
                assert False
            if neg:
                arg_num *= -1
        end_expr, expr_val = interp.next_op(op_name, arg_num)
        trace.append((op_name, arg_num))
        if end_expr:
            ds.append(state.s.output())
            expr_vals.append(expr_val)
            if op_name == 'exit':
                break
            interp = Interpreter()
        state = state.next_state(expr_val, op_name, arg_ref)
    answer = state.predict_answer().npvalue().argmax()
    return trace, expr_vals, answer


def solve2(encoder, decoder, raw_question, raw_options, max_op_count):
    question = parse_question(raw_question)
    options = [parse_question(raw_option) for raw_option in raw_options]
    input_nums = build_index(question, options)
    input_nums_pos = defaultdict(set)
    for index, num in input_nums.items():
        input_nums_pos[num].add(index)
    expr_nums_pos = defaultdict(set)
    num_candidates = set(input_nums_pos.keys()) | set(decoder.prior_nums.keys())
    num_candidates.remove(UNK)
    num_candidates |= {-num for num in num_candidates}
    num_candidates = {float(num) for num in num_candidates}
    dy.renew_cg()
    es, e, option_embeds = encoder(process_words(question), map(process_words, options))
    state = decoder(es, e, option_embeds, input_nums.keys())
    interp = Interpreter()
    expr_vals = []
    ds = []
    trace = []
    for _ in range(max_op_count):
        p_op = dy.log(state.op_probs()).npvalue()
        p_op[[op_name not in interp.valid_ops for op_id, op_name in decoder.opid2name.items()]] = -np.inf
        op_name = decoder.opid2name[p_op.argmax()]
        max_arg_num = None
        max_arg_ref = None
        max_instruct_p = None
        if op_name == 'load':
            load_p = p_op[decoder.name2opid['load']]
            copy_p = state.copy_probs()
            for arg_num in num_candidates:
                from_pos_prior_p, pos_prior_ref = state.from_prior_prob(arg_num)
                from_neg_prior_p, neg_prior_ref = state.from_prior_prob(-arg_num, True)
                from_pos_input_p, pos_input_ref = state.from_input_prob(input_nums_pos[arg_num])
                from_neg_input_p, neg_input_ref = state.from_input_prob(input_nums_pos[-arg_num], True)
                from_pos_exprs_p, pos_exprs_ref = state.from_exprs_prob(expr_nums_pos[arg_num])
                from_neg_exprs_p, neg_exprs_ref = state.from_exprs_prob(expr_nums_pos[-arg_num], True)

                from_p = dy.concatenate([from_pos_prior_p, from_neg_prior_p,
                                         from_pos_input_p, from_neg_input_p,
                                         from_pos_exprs_p, from_neg_exprs_p])
                arg_ref = (dy.concatenate_cols([pos_prior_ref, neg_prior_ref,
                                               pos_input_ref, neg_input_ref,
                                               pos_exprs_ref, neg_exprs_ref]) * copy_p)
                instruct_p = load_p + dy.log(dy.dot_product(copy_p, from_p)).value()
                if max_instruct_p is None or max_instruct_p < instruct_p:
                    max_arg_num = arg_num
                    max_arg_ref = arg_ref
                    max_instruct_p = instruct_p
        end_expr, expr_val = interp.next_op(op_name, max_arg_num)
        trace.append((op_name, max_arg_num))
        if end_expr:
            if op_name == 'exit':
                break
            ds.append(state.s.output())
            expr_val = float(expr_val)
            expr_nums_pos[expr_val].add(state.step)
            num_candidates.add(expr_val)
            num_candidates.add(-expr_val)
            expr_vals.append(expr_val)
            interp = Interpreter()
        state = state.next_state(expr_val, op_name, max_arg_ref)
    answer = state.predict_answer().npvalue().argmax()
    return trace, expr_vals, answer


def solve3(encoder, decoder, raw_question, raw_options, max_op_count, k, max_only=False):
    question = parse_question(raw_question)
    options = [parse_question(raw_option) for raw_option in raw_options]
    input_nums = build_index(question, options)
    input_nums_pos = defaultdict(set)
    for index, num in input_nums.items():
        input_nums_pos[num].add(index)
    num_candidates = set(input_nums_pos.keys()) | set(decoder.prior_nums.keys())
    num_candidates.remove(UNK)
    num_candidates |= {-num for num in num_candidates}
    num_candidates = {float(num) for num in num_candidates}
    dy.renew_cg()
    es, e, option_embeds = encoder(process_words(question), map(process_words, options))
    state = decoder(es, e, option_embeds, input_nums.keys())
    interp = Interpreter()

    def get_all_next(total_p, _, state, interp, last_op_name, last_arg_ref, last_arg_num, num_candidates, expr_nums_pos, expr_vals, trace):
        if last_op_name == 'exit':
            return
        if last_op_name is not None:
            trace = trace + [(last_op_name, last_arg_num)]
            interp = Interpreter(interp)
            end_expr, expr_val = interp.next_op(last_op_name, last_arg_num)
            if end_expr:
                try:
                    expr_val = float(expr_val)
                except:
                    return
                expr_nums_pos = defaultdict(set, expr_nums_pos)
                expr_nums_pos[expr_val].add(state.step)
                num_candidates = num_candidates | {expr_val, -expr_val}
                expr_vals = expr_vals + [expr_val]
                interp = Interpreter()
            state = state.next_state(expr_val, last_op_name, last_arg_ref)
        p_op = dy.log(state.op_probs()).npvalue()
        for op_id, op_name in decoder.opid2name.items():
            if op_name not in interp.valid_ops:
                continue
            op_p = p_op[op_id]
            if op_name == 'load':
                copy_p = state.copy_probs()
                for arg_num in num_candidates:
                    from_pos_prior_p, pos_prior_ref = state.from_prior_prob(arg_num)
                    from_neg_prior_p, neg_prior_ref = state.from_prior_prob(-arg_num, True)
                    from_pos_input_p, pos_input_ref = state.from_input_prob(input_nums_pos[arg_num])
                    from_neg_input_p, neg_input_ref = state.from_input_prob(input_nums_pos[-arg_num], True)
                    from_pos_exprs_p, pos_exprs_ref = state.from_exprs_prob(expr_nums_pos[arg_num])
                    from_neg_exprs_p, neg_exprs_ref = state.from_exprs_prob(expr_nums_pos[-arg_num], True)

                    from_p = dy.concatenate([from_pos_prior_p, from_neg_prior_p,
                                             from_pos_input_p, from_neg_input_p,
                                             from_pos_exprs_p, from_neg_exprs_p])
                    arg_ref = (dy.concatenate_cols([pos_prior_ref, neg_prior_ref,
                                                   pos_input_ref, neg_input_ref,
                                                   pos_exprs_ref, neg_exprs_ref]) * copy_p)
                    instruct_p = (op_p + dy.log(dy.dot_product(copy_p, from_p))).value()
                    if not math.isinf(instruct_p):
                        yield total_p + instruct_p, np.random.uniform(), state, interp, op_name, arg_ref, arg_num, num_candidates, expr_nums_pos, expr_vals, trace
            else:
                instruct_p = op_p
                yield total_p + instruct_p, np.random.uniform(), state, interp, op_name, None, None, num_candidates, expr_nums_pos, expr_vals, trace

    top_n = [[0.0, np.random.uniform(), state, interp, None, None, None, num_candidates, defaultdict(set), [], []]]
    final_c = []
    while len(final_c) < k:
        next_top_n = []
        for c in top_n:
            for next_c in get_all_next(*c):
                if next_c[4] == 'exit':
                    if len(final_c) < k:
                        heapq.heappush(final_c, next_c)
                    else:
                        heapq.heappushpop(final_c, next_c)
                    continue
                if len(next_c[-1]) > max_op_count:
                    continue
                if len(next_top_n) < k:
                    try:
                        heapq.heappush(next_top_n, next_c)
                    except:
                        print(next_top_n, next_c)
                        raise
                else:
                    try:
                        heapq.heappushpop(next_top_n, next_c)
                    except:
                        print(next_top_n, next_c)
                        raise
        top_n = next_top_n

    if max_only:
        final_p = max(final_c)[2].predict_answer().npvalue()
    else:
        final_p = None
        for c in final_c:
            if final_p is None:
                final_p = (c[0] + dy.log(c[2].predict_answer())).npvalue()
            else:
                final_p = np.logaddexp(final_p, (c[0] + dy.log(c[2].predict_answer())).npvalue())
    return final_p.argmax()


def cal_loss(encoder, decoder, question, options, input_num_indexes, trace, answer):
    es, e, option_embeds = encoder(question, options)
    state = decoder(es, e, option_embeds, input_num_indexes)
    problem_losses = []
    for instruction in trace:
        op_name = instruction[0]
        expr_val = instruction[1]
        item_loss = -dy.log(state.op_prob(op_name))
        arg_num = None
        if len(instruction) > 2:
            _, _, pos_prior_nid, neg_prior_nid, pos_input_indexes, neg_input_indexes, pos_exprs_indexes, \
            neg_exprs_indexes, arg_num = instruction
            copy_p = state.copy_probs()
            from_pos_prior_p, pos_prior_ref = state.from_prior_prob(pos_prior_nid)
            from_neg_prior_p, neg_prior_ref = state.from_prior_prob(neg_prior_nid, True)
            from_pos_input_p, pos_input_ref = state.from_input_prob(set(pos_input_indexes))
            from_neg_input_p, neg_input_ref = state.from_input_prob(set(neg_input_indexes), True)
            from_pos_exprs_p, pos_exprs_ref = state.from_exprs_prob(set(pos_exprs_indexes))
            from_neg_exprs_p, neg_exprs_ref = state.from_exprs_prob(set(neg_exprs_indexes), True)

            from_p = dy.concatenate([from_pos_prior_p, from_neg_prior_p,
                                     from_pos_input_p, from_neg_input_p,
                                     from_pos_exprs_p, from_neg_exprs_p])
            item_loss += -dy.log(dy.dot_product(copy_p, from_p))
            arg_ref = dy.concatenate_cols([pos_prior_ref, neg_prior_ref,
                                           pos_input_ref, neg_input_ref,
                                           pos_exprs_ref, neg_exprs_ref]) * copy_p
        problem_losses.append(item_loss)
        if op_name == 'exit':
            break
        state = state.next_state(expr_val, op_name, arg_ref)
    answer_loss = -dy.log(dy.pick(state.predict_answer(), answer))
    problem_losses.append(answer_loss)
    return dy.esum(problem_losses)


def main():
    parser = argparse.ArgumentParser(description='Train attention model')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--vocab_file', default='./vocab.dmp', type=str)
    parser.add_argument('--train_set', default='./train_set.dmp', type=str)
    parser.add_argument('--valid_set', default='./valid_set.dmp', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--trainer', default='adam', choices={'sgd', 'adam', 'adagrad'}, type=str)
    parser.add_argument('--word_embed_dim', default=256, type=int)
    parser.add_argument('--encoder_num_layers', default=2, type=int)
    parser.add_argument('--encoder_state_dim', default=256, type=int)
    parser.add_argument('--op_embed_dim', default=32, type=int)
    parser.add_argument('--num_embed_dim', default=256, type=int)
    parser.add_argument('--sign_embed_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=128, type=int)
    parser.add_argument('--decoder_num_layers', default=2, type=int)
    parser.add_argument('--decoder_state_dim', default=256, type=int)
    parser.add_argument('--dropout', default=None, type=float)
    parser.add_argument('--seed', default=11747, type=int)
    parser.add_argument('--max_op_count', default=50, type=int)

    args, _ = parser.parse_known_args()

    with open(args.vocab_file, 'rb') as f:
        op_names, word2wid, wid2word, num2nid, nid2num = pickle.load(f)
        op_names = sorted(op_names)

    with open(args.train_set, 'rb') as f:
        train_set = pickle.load(f)

    if len(train_set) > 0 and len(train_set[0][2][0]) == 8:
        print('add expr values...')
        train_set = add_expr_val(train_set)
        with open(args.train_set, 'wb') as f:
            pickle.dump(train_set, f)

    if len(train_set) > 0 and type(train_set[0][0][0]) == str:
        print('add num values...')
        train_set = add_num_val(train_set)
        with open(args.train_set, 'wb') as f:
            pickle.dump(train_set, f)

    print('size of train_set:', len(train_set))

    random.seed(args.seed)

    model = dy.ParameterCollection()

    if args.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(model)
    elif args.trainer == 'adam':
        trainer = dy.AdamTrainer(model)
    elif args.trainer == 'adagrad':
        trainer = dy.AdagradTrainer(model)

    encoder = Encoder(model, word2wid, args.word_embed_dim, args.encoder_num_layers, args.encoder_state_dim)
    decoder = Decoder(model, op_names, args.op_embed_dim, num2nid, args.num_embed_dim, args.sign_embed_dim,
                      args.encoder_state_dim, args.att_dim, args.decoder_num_layers, args.decoder_state_dim)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.model_path is None:
        model.save('%s/init.dmp' % args.checkpoint_dir)
    else:
        model.populate(args.model_path)

    if args.dropout is not None:
        encoder.set_dropout(args.dropout)
        decoder.set_dropout(args.dropout)

    num_problems = len(train_set)
    for num_epoch in itertools.count(1):
        random.shuffle(train_set)
        epoch_loss = 0.0
        epoch_seq_length = 0
        batch_losses = []
        batch_seq_length = 0
        num_batch = 0
        dy.renew_cg()
        for i, (question, options, trace, input_num_indexes, answer) in enumerate(train_set, 1):
            problem_loss = cal_loss(encoder, decoder, question, options, input_num_indexes, trace, answer)
            batch_losses.append(problem_loss)
            batch_seq_length += len(trace)
            epoch_seq_length += len(trace)
            if i % args.batch_size == 0 or i == num_problems:
                batch_loss = dy.esum(batch_losses) / len(batch_losses)
                batch_loss.backward()
                trainer.update()
                batch_loss_value = batch_loss.value()
                batch_per_item_loss = batch_loss_value / batch_seq_length
                epoch_loss += batch_loss_value
                epoch_perplexity = math.exp(epoch_loss / epoch_seq_length)
                dy.renew_cg()
                num_batch += 1
                batch_losses = []
                batch_seq_length = 0
                if num_batch % 20 == 0:
                    print('epoch %d, batch %d, batch_per_item_loss %f, epoch_perplexity %f' % \
                          (num_epoch, num_batch, batch_per_item_loss, epoch_perplexity))
        model.save('%s/epoch_%d.dmp' % (args.checkpoint_dir, num_epoch))


# python stack_machine_model.py --dynet-seed 11747 --dynet-autobatch 1 --dynet-mem 2000 --dynet-gpu --batch_size 64 --dropout 0.7
if __name__ == "__main__":
    main()
