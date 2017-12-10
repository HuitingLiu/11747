# coding: utf-8

import dynet as dy
import numpy as np
import argparse
import pickle
import itertools
import random
import math
import os

from contextlib import contextmanager
from collections import Iterable
from itertools import chain
from sympy.parsing.sympy_parser import parse_expr
from interpreter import Interpreter
from preprocessing.parser import parse_question, extract_instructions

UNK = ('<UNK>',)


@contextmanager
def parameters(*params):
    yield tuple(map(lambda x: dy.parameter(x), params))


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

        self.fwdLSTM = dy.LSTMBuilder(num_layers, word_embed_dim, hidden_dim / 2, model)
        self.bwdLSTM = dy.LSTMBuilder(num_layers, word_embed_dim, hidden_dim / 2, model)

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

    def __call__(self, question, options):
        q_seq = self.words_embeds(question)

        fqs = self.fwdLSTM.initial_state().add_inputs(q_seq)
        bqs = self.bwdLSTM.initial_state().add_inputs(reversed(q_seq))
        es = [dy.concatenate([f.output(), b.output()]) for f, b in zip(fqs, reversed(bqs))]

        s = [dy.concatenate([x, y]) for x, y in zip(fqs[-1].s(), bqs[-1].s())]

        for option in options:
            o_seq = self.words_embeds(option)
            fos = fqs[-1].transduce(o_seq)
            bos = bqs[-1].transduce(reversed(o_seq))
            es.extend(dy.concatenate([f, b]) for f, b in zip(fos, reversed(bos)))

        return es, s


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
                es_matrix = dy.concatenate_cols(es)
                ps_matrix = P * es_matrix

                def cal_scores(s):
                    hs_matrix = dy.tanh(dy.colwise_add(ps_matrix, W * s))
                    return dy.softmax(dy.transpose(b * hs_matrix))

                def cal_context(s, selected=None):
                    ws = cal_scores(s)
                    if selected is None:
                        return es_matrix * ws, ws
                    selected_ws = dy.select_rows(ws, selected)
                    selected_ws = dy.cdiv(selected_ws, dy.sum_elems(selected_ws))
                    return dy.concatenate_cols([es[index] for index in selected]) * selected_ws, ws

                return cal_scores, cal_context, None
            else:
                es = []
                ps = []

                def append_e(e):
                    es.append(e)
                    ps.append(P * e)

                def cal_scores(s):
                    if len(ps) == 0:
                        return None
                    hs_matrix = dy.tanh(dy.colwise_add(dy.concatenate_cols(ps), W * s))
                    return dy.softmax(dy.transpose(b * hs_matrix))

                def cal_context(s, selected=None):
                    ws = cal_scores(s)
                    if ws is None:
                        return None, None
                    if selected is None:
                        return dy.concatenate_cols(es) * ws, ws
                    selected_ws = dy.select_rows(ws, selected)
                    selected_ws = dy.cdiv(selected_ws, dy.sum_elems(selected_ws))
                    return dy.concatenate_cols([es[index] for index in selected]) * selected_ws, ws

                return cal_scores, cal_context, append_e


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
        self.dummy_arg_dim = num_embed_dim + sign_embed_dim

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

        self.h2op = Linear(model, hidden_dim, num_ops)
        self.h2ht = Linear(model, hidden_dim, hidden_dim)
        self.h2copy = Linear(model, hidden_dim, 6)

        num_prior = len(prior_nums)
        self.sh2prior = Linear(model, hidden_dim + sign_embed_dim, num_prior)

        self.opLSTM = dy.LSTMBuilder(num_layers, encode_dim + op_embed_dim + num_embed_dim + sign_embed_dim, hidden_dim, model)

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

    def __call__(self, es, e, input_num_indexes):
        assert isinstance(input_num_indexes, Iterable)

        _, cal_context, _ = self.decode_att(es)
        input_num_indexes = sorted(set(input_num_indexes))
        cal_input_scores, cal_input_context, _ = self.input_att([es[index] for index in input_num_indexes])
        exprs_num_indexs = []
        cal_exprs_scores, cal_exprs_context, append_expr = self.exprs_att()
        s = [self.opLSTM.initial_state().set_s(e)]
        step = [-1]

        def op_probs():
            h = s[0].output()
            return dy.softmax(self.h2op(h))

        def op_prob(op):
            probs = op_probs()
            if type(op) == str:
                op = self.name2opid[op]
            return dy.pick(probs, op)

        def copy_probs():
            h = s[0].output()
            return dy.softmax(self.h2copy(h))

        def _signed_h(h=None, neg=False):
            with parameters(self.neg_embed, self.pos_embed) as (neg_embed, pos_embed):
                signed_h = dy.concatenate([h if h is not None else s[0].output(), neg_embed if neg else pos_embed])
                return signed_h

        def from_prior_probs(neg=False):
            signed_h = _signed_h(neg=neg)
            return dy.softmax(self.sh2prior(signed_h))

        def from_prior_prob(num, neg=False):
            probs = from_prior_probs(neg)
            if type(num) == float:
                num = self.name2opid[num]
            with parameters(self.neg_prior_embed, self.pos_prior_embed) as (neg_prior_embed, pos_prior_embed):
                prior_ref = dy.concatenate([self.num_embeds[num], neg_prior_embed if neg else pos_prior_embed])
            return dy.pick(probs, num), prior_ref

        def from_input_probs(neg=False):
            signed_h = _signed_h(neg=neg)
            return cal_input_scores(signed_h)

        def from_input_prob(selected_indexes, neg=False):
            assert type(selected_indexes) == set
            selected_indexes = [index for index, old_index in enumerate(input_num_indexes) if old_index in selected_indexes]
            if len(selected_indexes) == 0:
                return dy.scalarInput(0), dy.zeros(self.dummy_arg_dim)

            signed_h = _signed_h(neg=neg)
            input_ref, probs = cal_input_context(signed_h, selected_indexes)
            with parameters(self.neg_input_embed, self.pos_input_embed) as (neg_input_embed, pos_input_embed):
                input_ref = dy.concatenate([input_ref, neg_input_embed if neg else pos_input_embed])
            return dy.sum_elems(dy.select_rows(probs, selected_indexes)), input_ref

        def from_exprs_probs(neg=False):
            ht = dy.tanh(self.h2ht(s[0].output()))
            signed_h = _signed_h(ht, neg)
            return cal_exprs_scores(signed_h)

        def from_exprs_prob(selected_indexes, neg=False):
            assert type(selected_indexes) == set
            selected_indexes = [index for index, old_index in enumerate(exprs_num_indexs) if old_index in selected_indexes]
            if len(selected_indexes) == 0:
                return dy.scalarInput(0), dy.zeros(self.dummy_arg_dim)

            ht = dy.tanh(self.h2ht(s[0].output()))
            signed_h = _signed_h(ht, neg)
            exprs_ref, probs = cal_exprs_context(signed_h, selected_indexes)
            with parameters(self.neg_exprs_embed, self.pos_exprs_embed) as (neg_exprs_embed, pos_exprs_embed):
                exprs_ref = dy.concatenate([exprs_ref, neg_exprs_embed if neg else pos_exprs_embed])
            return dy.sum_elems(dy.select_rows(probs, selected_indexes)), exprs_ref

        with parameters(self.start_op, self.dummy_arg) as (start_op, dummy_arg):
            def next_state(op, arg_ref=None):
                op_embed = self.op_embeds[op] if type(op) in (int, str) else op
                arg_embed = dummy_arg if arg_ref is None else arg_ref
                context, _ = cal_context(s[0].output())
                s[0] = s[0].add_input(dy.concatenate([context, op_embed, arg_embed]))
                if op in ('end', self.name2opid['end']):
                    append_expr(s[0].output())
                    exprs_num_indexs.append(step[0])
                step[0] += 1
                return s[0].output()

            next_state(start_op)
            step[0] = 0

        return op_probs, op_prob, copy_probs, from_prior_probs, from_prior_prob, from_input_probs, \
               from_input_prob, from_exprs_probs, from_exprs_prob, next_state


def find_num_positions(dataset):
    result = []
    for question, options, trace in dataset:
        input_num_indexes = set()
        for index, token in enumerate(itertools.chain(*([question] + options))):
            try:
                if parse_expr(token).is_number:
                    input_num_indexes.add(index)
            except:
                pass
        result.append((question, options, trace, input_num_indexes))
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
    es, e = encoder(question, options)
    op_probs, _, copy_probs, from_prior_probs, _, from_input_probs, _, from_exprs_probs, _, next_state \
        = decoder(es, e, input_nums.keys())
    interp = Interpreter()
    expr_vals = []
    ds = []
    for _ in range(max_op_count):
        p_op = op_probs().npvalue()
        p_op[[op_name not in interp.valid_ops for op_id, op_name in decoder.opid2name.items()]] = -np.inf
        op_name = decoder.opid2name[p_op.argmax()]
        arg_num = None
        arg_ref = None
        if op_name == 'load':
            copy_src = copy_id2src[copy_probs().npvalue().argmax()]
            neg = copy_src.startswith('neg')
            if copy_src.endswith('_input'):
                num_index = input_nums_indexes[from_input_probs(neg=neg).npvalue().argmax()]
                arg_num = input_nums[num_index]
                with parameters(decoder.neg_input_embed, decoder.pos_input_embed) as (neg_input_embed, pos_input_embed):
                    arg_ref = dy.concatenate([es[num_index], neg_input_embed if neg else pos_input_embed])
            elif copy_src.endswith('_prior'):
                p_prior = from_prior_probs(neg=neg).npvalue()
                p_prior[UNK_id] = -np.inf
                nid = p_prior.argmax()
                arg_num = decoder.nid2num[nid]
                with parameters(decoder.neg_prior_embed, decoder.pos_prior_embed) as (neg_prior_embed, pos_prior_embed):
                    arg_ref = dy.concatenate([decoder.num_embeds[nid], neg_prior_embed if neg else pos_prior_embed])
            elif copy_src.endswith('_exprs'):
                expr_index = from_exprs_probs(neg=neg).npvalue().argmax()
                arg_num = expr_vals[expr_index]
                with parameters(decoder.neg_exprs_embed, decoder.pos_exprs_embed) as (neg_exprs_embed, pos_exprs_embed):
                    arg_ref = dy.concatenate([ds[expr_index], neg_exprs_embed if neg else pos_exprs_embed])
            else:
                assert False
            if neg:
                arg_num *= -1
        dh = next_state(op_name, arg_num)
        end_expr, expr_val = interp.next_op(op_name, arg_num)
        if end_expr:
            ds.append(dh)
            expr_vals.append(expr_val)
            if op_name == 'exit':
                break
            interp = Interpreter()
    return expr_vals


def cal_loss(encoder, decoder, question, options, input_num_indexes, trace):
    es, e = encoder(question, options)
    _, op_prob, copy_probs, _, from_prior_prob, _, from_input_prob, _, from_exprs_prob, next_state \
        = decoder(es, e, input_num_indexes)
    problem_losses = []
    for instruction in trace:
        op_name = instruction[0]
        item_loss = -dy.log(op_prob(op_name))
        if len(instruction) > 1:
            _, pos_prior_nid, neg_prior_nid, pos_input_indexes, neg_input_indexes, pos_exprs_indexes, \
            neg_exprs_indexes = instruction
            copy_p = copy_probs()
            from_pos_prior_p, pos_prior_ref = from_prior_prob(pos_prior_nid)
            from_neg_prior_p, neg_prior_ref = from_prior_prob(neg_prior_nid, True)
            from_pos_input_p, pos_input_ref = from_input_prob(set(pos_input_indexes))
            from_neg_input_p, neg_input_ref = from_input_prob(set(neg_input_indexes), True)
            from_pos_exprs_p, pos_exprs_ref = from_exprs_prob(set(pos_exprs_indexes))
            from_neg_exprs_p, neg_exprs_ref = from_exprs_prob(set(neg_exprs_indexes), True)

            from_p = dy.concatenate([from_pos_prior_p, from_neg_prior_p,
                                     from_pos_input_p, from_neg_input_p,
                                     from_pos_exprs_p, from_neg_exprs_p])
            item_loss += -dy.log(dy.dot_product(copy_p, from_p))
            arg_ref = dy.concatenate_cols([pos_prior_ref, neg_prior_ref,
                                           pos_input_ref, neg_input_ref,
                                           pos_exprs_ref, neg_exprs_ref]) * copy_p
            next_state(op_name, arg_ref)
        else:
            next_state(op_name)
        problem_losses.append(item_loss)
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

    if len(train_set) > 0 and len(train_set[0]) == 3:
        train_set = find_num_positions(train_set)
        with open(args.train_set, 'wb') as f:
            pickle.dump(train_set, f)

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
        for i, (question, options, trace, input_num_indexes) in enumerate(train_set, 1):
            problem_loss = cal_loss(encoder, decoder, question, options, input_num_indexes, trace)
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


# python stack_machine_model.py --dynet-seed 11747 --dynet-autobatch 1 --dynet-mem 11000 --dynet-gpu --batch_size 64 --dropout 0.7
if __name__ == "__main__":
    main()
