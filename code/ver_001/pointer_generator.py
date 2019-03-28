import torch
import os
import argparse
import time
import sys
import logging
from easydict import EasyDict
import random
import numpy as np
from torch.utils.data import DataLoader
import json
import progressbar
progressbar.streams.flush()
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from collections import namedtuple

def main():
    G = EasyDict()
    G.flags = get_flags()
    G.logger = get_logger()
    G.logger.info('Program start in "{m}" mode'.format(m=G.flags.mode))
    G.vocab = Vocabulary(G)

    G.logger.info('Program running with following args:')
    for i in ['{k} = {v}'.format(k=item[0], v=item[1])
              for item in sorted(list(vars(G.flags).items()), key=lambda item: item[0])]:
        G.logger.info('  ' + i)

    G.device = torch.device('cuda')
    random.seed(G.flags.seed)
    np.random.seed(G.flags.seed)
    torch.manual_seed(G.flags.seed)

    start_epoch = 0
    start_iter = 0

    model = PointerGeneratorNetwork(G)
    # TODO: support reload weights?

    model.to(G.device)

    if G.flags.mode == 'train':
        conduct_train(G, model, start_epoch, start_iter)
    else:
        raise ValueError('Invalid mode {m}. (Only support modes: train)'.format(m=G.flags.mode))


def conduct_train(G, model, start_epoch, start_iter):
    print('train....')
    # dataset_train = DocumentDataset(G, 'training')
    # dataset_val = DocumentDataset(G, 'validation')
    dataset_test = DocumentDataset(G, 'testing')
    dataset_train = dataset_test
    dataset_valid = dataset_test

    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=G.flags.train_batch_size)
    data_loader_valid = DataLoader(dataset_valid, shuffle=False, batch_size=1)  # batch_size=1, needs further optimize
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=1)

    G.logger.info('Conduct training:')
    G.logger.info('    Batch Size = {bs}'.format(bs=G.flags.train_batch_size))
    G.logger.info('    Max Epochs = {epoch}'.format(epoch=G.flags.train_epoch))

    param_optimizer = list(model.named_parameters())
    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=G.flags.train_learning_rate, betas=(0.8, 0.999))

    G.logger.info('Evaluation on Random Initialized Parameters')
    conduct_evaluation(G, model, data_loader_valid, data_loader_test)

    # TODO: optimize for reloadable training
    global_step = 0
    best_eval_loss = -np.inf
    for epoch in range(1, G.flags.train_epoch + 1):
        G.logger.info('Epoch {e} running:'.format(e=epoch))
        train_progress = tqdm.trange(len(data_loader_train), desc='Train', leave=True)
        model.train()
        epoch_loss_list = []

        for idx, batch_data in enumerate(data_loader_train):
            batch_data = [item.to(G.device) for item in batch_data]
            loss = model(*batch_data)

            train_progress.update()
        train_progress.close()

    pass

def conduct_evaluation(G, model, data_loader_valid, data_loader_test):
    pass


def get_logger(thread_id=0):
    logger = logging.getLogger('Thread {tid}'.format(tid=thread_id))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='Running mode: train|')
    parser.add_argument('--seed', type=int, default=20190323)

    parser.add_argument('--vocab_size', type=int, default=50000, help='default max vocabulary size, other words will be regard OOV words')
    parser.add_argument('--dataset_vocab_file_path', type=str, default='/s1_md0/v-botsh/Research/Projects/pointer_generator.pytorch/datasets/vocab')
    parser.add_argument('--dataset_training_file_path', type=str, default='/s1_md0/v-botsh/Research/Projects/pointer_generator.pytorch/datasets/train.json')
    parser.add_argument('--dataset_validation_file_path', type=str, default='/s1_md0/v-botsh/Research/Projects/pointer_generator.pytorch/datasets/val.json')
    parser.add_argument('--dataset_testing_file_path', type=str, default='/s1_md0/v-botsh/Research/Projects/pointer_generator.pytorch/datasets/test.json')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_epoch', type=int, default=1000)

    parser.add_argument('--max_decoder_length', type=int, default=100)
    parser.add_argument('--max_encoder_length', type=int, default=400)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--rnn_hidden_dim', type=int, default=256)
    parser.add_argument('--rnn_dropout', type=float, default=0.5)

    args = parser.parse_args()
    return args


class DocumentDataset(torch.utils.data.Dataset):
    # use article to construct OOV list first
    # [UNK](0) is allowed to encoder_original_vocab_input and decoder_original_input
    # use OOV idx instead for encoder_extended_vocab_input and decoder_extended_vocab_target

    # decoder parts will be useless when inferring
    sample_item_fields = ['encoder_original_vocab_input', 'encoder_extended_vocab_input', 'encoder_mask', 'encoder_length',
                            'decoder_original_vocab_input', 'decoder_extended_vocab_target', 'decoder_mask', 'decoder_length',
                            'original_article', 'original_abstract_sentence_list', 'original_abstract',
                            'oov_idx_to_token', 'oov_token_to_idx']
    SampleItem = namedtuple('SampleItem', sample_item_fields)

    def __init__(self, G, data_partition):
        super(DocumentDataset, self).__init__()
        G.logger.info('Loading {p} data'.format(p=data_partition))
        if data_partition not in ['training', 'validation', 'testing']: raise ValueError('data partition is not supported')
        # only training stage needs decoder_target
        self._has_decoder_target = True if data_partition in ['training'] else False
        self.max_decoder_length = G.flags.max_decoder_length
        self.max_encoder_length = G.flags.max_encoder_length

        self._sample_list = []
        self._dataset_file_path = {
            'training': G.flags.dataset_training_file_path,
            'validation': G.flags.dataset_validation_file_path,
            'testing': G.flags.dataset_testing_file_path
        }[data_partition]
        self._dataset_item_list = json.load(open(self._dataset_file_path))


        for dataset_item in progressbar.progressbar(self._dataset_item_list):
            article_text = dataset_item['article']
            abstract_text = dataset_item['abstract']

            ################################
            # process article_first
            ################################
            encoder_oov_idx_to_token = []
            encoder_oov_token_to_idx = dict()
            encoder_original_vocab_input = []
            encoder_extended_vocab_input = []
            encoder_mask = []
            article_token_list = article_text.split(' ')[:self.max_encoder_length]
            encoder_length = len(article_token_list)
            for token in article_token_list:
                if token in G.vocab.token_to_idx:
                    encoder_original_vocab_input.append(G.vocab.token_to_idx[token])
                    encoder_extended_vocab_input.append(G.vocab.token_to_idx[token])
                else:
                    if token not in encoder_oov_idx_to_token:
                        encoder_oov_token_to_idx[token] = G.vocab.vocab_size + len(encoder_oov_idx_to_token)
                        encoder_oov_idx_to_token.append(token)
                    encoder_original_vocab_input.append(G.vocab.token_to_idx[Vocabulary.UNK_TOKEN])
                    encoder_extended_vocab_input.append(encoder_oov_token_to_idx[token])
                encoder_mask.append(1)

            encoder_pad_count = self.max_encoder_length - encoder_length
            encoder_original_vocab_input = encoder_original_vocab_input + [G.vocab.token_to_idx[Vocabulary.PAD_TOKEN]] * encoder_pad_count
            encoder_extended_vocab_input = encoder_extended_vocab_input + [G.vocab.token_to_idx[Vocabulary.PAD_TOKEN]] * encoder_pad_count
            encoder_mask = encoder_mask + [0] * encoder_pad_count
            original_article = article_text

            ################################
            # process abstact then
            ################################
            # split abstract_text to a list of sentences
            abstract_sentence_list = [item.strip() for item in [item.split(Vocabulary.ABSTRACT_SENTENCE_END_TOKEN)[0] for item in abstract_text.split(Vocabulary.ABSTRACT_SENTENCE_START_TOKEN)[1:]]]
            abstract_text = ' '.join(abstract_sentence_list)
            abstract_token_list = abstract_text.split(' ')[:self.max_decoder_length - 1]  # extra 1 for [BOS] or [EOS]
            decoder_length = len(abstract_token_list) + 1
            decoder_original_input = [G.vocab.token_to_idx[Vocabulary.DECODING_BOS_TOKEN]]
            decoder_extended_vocab_target = []
            decoder_mask = [1]
            for token in abstract_token_list:
                token_idx = G.vocab.token_to_idx.get(token, G.vocab.token_to_idx[Vocabulary.UNK_TOKEN])
                decoder_original_input.append(token_idx)
                if token_idx == G.vocab.token_to_idx[Vocabulary.UNK_TOKEN]:
                    decoder_extended_vocab_target.append(encoder_oov_token_to_idx.get(token, G.vocab.token_to_idx[Vocabulary.UNK_TOKEN]))
                else:
                    decoder_extended_vocab_target.append(token_idx)
                decoder_mask.append(1)
            decoder_extended_vocab_target.append(G.vocab.token_to_idx[Vocabulary.DECODING_EOS_TOKEN])

            decoder_pad_count = self.max_decoder_length - decoder_length
            decoder_original_input = decoder_original_input + [G.vocab.token_to_idx[Vocabulary.PAD_TOKEN]] * decoder_pad_count
            decoder_extended_vocab_target = decoder_extended_vocab_target + [G.vocab.token_to_idx[Vocabulary.PAD_TOKEN]] * decoder_pad_count
            decoder_mask = decoder_mask + [0] * decoder_pad_count
            original_abstract_sentence_list = abstract_sentence_list
            original_abstract = abstract_text

            self._sample_list.append(self.SampleItem(
                encoder_original_vocab_input=np.asarray(encoder_original_vocab_input),
                encoder_extended_vocab_input=np.asarray(encoder_extended_vocab_input),
                encoder_mask=np.asarray(encoder_mask),
                encoder_length=encoder_length,
                decoder_original_vocab_input=np.asarray(decoder_original_input),
                decoder_extended_vocab_target=np.asarray(decoder_extended_vocab_target),
                decoder_mask=np.asarray(decoder_mask),
                decoder_length=decoder_length,
                original_article=original_article,
                original_abstract_sentence_list=original_abstract_sentence_list,
                original_abstract=original_abstract,
                oov_idx_to_token=encoder_oov_idx_to_token,
                oov_token_to_idx=encoder_oov_token_to_idx
            ))

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, idx):
        item = self._sample_list[idx]
        return \
            torch.Tensor(item.encoder_original_vocab_input).long(), \
            torch.Tensor(item.encoder_extended_vocab_input).long(), \
            torch.Tensor(item.encoder_mask).float(), \
            torch.Tensor(item.decoder_original_vocab_input).long(), \
            torch.Tensor(item.decoder_extended_vocab_target).long(), \
            torch.Tensor(item.decoder_mask).float(), \
            torch.Tensor([len(item.oov_token_to_idx)]).long()  # oov count


class Vocabulary:
    ABSTRACT_SENTENCE_START_TOKEN = '<s>'
    ABSTRACT_SENTENCE_END_TOKEN = '</s>'
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    DECODING_BOS_TOKEN = '[BOS]'
    DECODING_EOS_TOKEN = '[EOS]'

    def __init__(self, G):
        self.G = G
        self._vocab_file_path = G.flags.dataset_vocab_file_path
        self._vocab_list = open(self._vocab_file_path, 'r').read().splitlines()
        self.idx_to_token = []
        self.idx_to_token.extend(
            [Vocabulary.UNK_TOKEN, Vocabulary.PAD_TOKEN, Vocabulary.DECODING_BOS_TOKEN, Vocabulary.DECODING_EOS_TOKEN]
        )
        self.idx_to_token.extend([item.split(' ')[0] for item in self._vocab_list][:G.flags.vocab_size - len(self.idx_to_token)])
        self.token_to_idx = dict([(token, idx) for idx, token in enumerate(self.idx_to_token)])
        self.vocab_size = len(self.idx_to_token)




class VarLenEncoder(torch.nn.Module):
    def __init__(self, G, embedding_layer, rnn_hidden_dim):
        super(VarLenEncoder, self).__init__()
        self.G = G
        self.embedding_dim = embedding_layer.embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.embedding_layer = embedding_layer
        self.encoder_rnn_cell = torch.nn.LSTM(self.embedding_dim, self.rnn_hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, encoder_original_vocab_input, encoder_mask):
        encoder_length = encoder_mask.sum(dim=-1)
        sorted_lengths, perm_index = encoder_length.sort(0, descending=True)
        sorted_encoder_input = encoder_original_vocab_input[perm_index]
        sorted_encoder_input = self.embedding_layer(sorted_encoder_input)
        packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_lengths.cpu().numpy(),
                                                    batch_first=True)
        packed_encoder_output, (sorted_encoder_h_t, sorted_encoder_c_t) = self.encoder_rnn_cell(packed_encoder_input)
        sorted_encoder_output, _ = pad_packed_sequence(packed_encoder_output, batch_first=True)
        _, unsorted_perm_index = perm_index.sort()
        # unsort all results
        encoder_output = sorted_encoder_output[unsorted_perm_index]
        encoder_h_t = sorted_encoder_h_t[:, unsorted_perm_index, :]
        encoder_c_t = sorted_encoder_c_t[:, unsorted_perm_index, :]

        return encoder_output, (encoder_h_t, encoder_c_t)

class StateReducer(torch.nn.Module):
    def __init__(self, G, hidden_size):
        super(StateReducer, self).__init__()
        self.G = G
        self.hidden_size = hidden_size
        self.reduce_c_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.reduce_h_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def forward(self, state_h_t, state_c_t):
        old_state_h_t = torch.cat(state_h_t.chunk(chunks=2, dim=0), dim=-1).squeeze(0)
        old_state_c_t = torch.cat(state_c_t.chunk(chunks=2, dim=0), dim=-1).squeeze(0)

        new_state_h_t = F.relu(self.reduce_h_layer(old_state_h_t))
        new_state_c_t = F.relu(self.reduce_h_layer(old_state_c_t))

        return new_state_h_t, new_state_c_t


class MaskedAttentionModule(torch.nn.Module):
    def __init__(self, G):
        super(MaskedAttentionModule, self).__init__()
        self.G = G

    def forward(self, e, encoder_mask):
        attention_distribution = F.softmax(e)
        attention_distribution *= encoder_mask
        masked_sums = attention_distribution.sum(dim=1)
        normalized_attention_distribution = attention_distribution / masked_sums.unsqueeze(-1)  # re-arrange to [0,1]
        return normalized_attention_distribution


class DecoderStepModule(torch.nn.Module):
    def __init__(self, G, v, attention_vector_size, decoder_rnn_hidden_dim, decoder_step_module_idx=-1, is_converage=False):
        super(DecoderStepModule, self).__init__()
        self.G = G
        self.v = v
        self.decoder_rnn_hidden_dim = decoder_rnn_hidden_dim
        self.attention_module_idx = decoder_step_module_idx
        self.is_converage = is_converage
        self.attention_vector_size = attention_vector_size
        self.linear_fusion_decoder_state = torch.nn.Linear(self.decoder_rnn_hidden_dim * 2, self.attention_vector_size)  # input = c_t and h_t
        self.masked_attention = MaskedAttentionModule(G)

    def forward(self, encoder_features, decoder_prev_step_state, encoder_state, encoder_mask, coverage=None):
        decoder_features = self.linear_fusion_decoder_state(torch.cat(decoder_prev_step_state, dim=1)).unsqueeze(1).unsqueeze(1)
        e = (self.v * F.tanh(encoder_features + decoder_features)).sum(dim=[2, 3])
        attention_distribution = self.masked_attention(e, encoder_mask)
        context_vector = (attention_distribution.unsqueeze(-1).unsqueeze(-1) * encoder_state).sum(dim=[1, 2])
        return context_vector, attention_distribution, coverage

class AttentionDecoder(torch.nn.Module):
    def __init__(self, G, embedding_layer, rnn_hidden_dim):
        super(AttentionDecoder, self).__init__()
        self.G = G
        self.embedding_dim = embedding_layer.embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.embedding_layer = embedding_layer
        self.decoder_rnn_cell = torch.nn.LSTMCell(self.embedding_dim, self.rnn_hidden_dim)
        self.attention_size = self.decoder_rnn_cell.hidden_size * 2 # *2 because of bidirectional encoder
        self.context_vector_size = self.attention_size
        self.attention_vector_size = self.attention_size + 100
        self.max_decoder_length = G.flags.max_decoder_length
        self.attention_h_conv = torch.nn.Conv2d(in_channels=self.attention_size, out_channels=self.attention_vector_size, kernel_size=1)
        self.v = torch.nn.Parameter(torch.randn([self.attention_vector_size]))  # randn is not same as official code, which uses xavier
        self.linear_fusion_input_with_context_vector = torch.nn.Linear(self.embedding_dim + self.context_vector_size, self.embedding_dim)
        # we need an attention model for each step, but for all steps (modules), the 'v' is shared.
        self.decoder_step_module_list = torch.nn.ModuleList([DecoderStepModule(G, self.v, self.attention_vector_size, self.decoder_rnn_cell.hidden_size, i) for i in range(self.max_decoder_length)])
        self.linear_fusion_context_vector_state_x = torch.nn.Linear(self.context_vector_size + self.decoder_rnn_cell.hidden_size * 2 + self.embedding_dim, 1)
        self.linear_fusion_output_context_vector = torch.nn.Linear(self.context_vector_size + self.decoder_rnn_cell.hidden_size, self.decoder_rnn_cell.hidden_size)

    def forward(self, decoder_original_vocab_input, encoder_output, reduced_encoder_state, encoder_mask):
        device = decoder_original_vocab_input.device
        batch_size = encoder_output.shape[0]
        encoder_features = encoder_output.transpose(2, 1).unsqueeze(-1)
        encoder_features = self.attention_h_conv(encoder_features)
        encoder_features = encoder_features.permute(0, 2, 3, 1)  # reshape: torch.BCHW->tf.BHWC

        encoder_state = encoder_output.unsqueeze(2)

        decoder_prev_step_state = reduced_encoder_state
        context_vector = torch.zeros([batch_size, self.context_vector_size]).to(device)  # zero for 0-th step
        decoder_input_embedded = self.embedding_layer(decoder_original_vocab_input)
        decoder_input_chunk_list = decoder_input_embedded.chunk(chunks=self.max_decoder_length, dim=1)

        output_list = []
        attention_distribution_list = []
        p_gen_list = []

        for idx, decoder_input_step in enumerate(decoder_input_chunk_list):
            decoder_input_step = decoder_input_step.squeeze(1)
            x = self.linear_fusion_input_with_context_vector(torch.cat([decoder_input_step, context_vector], dim=-1))
            decoder_prev_step_state = self.decoder_rnn_cell(x, decoder_prev_step_state)
            context_vector, attention_distribution, _ = self.decoder_step_module_list[idx](encoder_features, decoder_prev_step_state, encoder_state, encoder_mask)
            attention_distribution_list.append(attention_distribution)

            # calculate p_gen
            p_gen = torch.cat([context_vector, decoder_prev_step_state[0], decoder_prev_step_state[1], x], dim=-1)
            p_gen = self.linear_fusion_context_vector_state_x(p_gen)
            p_gen = F.sigmoid(p_gen)
            p_gen_list.append(p_gen)

            # calculate output
            cell_output = decoder_prev_step_state[0]  # h_t
            output = self.linear_fusion_output_context_vector(torch.cat([cell_output, context_vector], dim=-1))
            output_list.append(output)

        # if converage is not None: xxx
        return output_list, decoder_prev_step_state, attention_distribution_list, p_gen_list, # converage




class PointerGeneratorDistributionModule(torch.nn.Module):

    def __init__(self, G, unextended_embedding_layer):
        super(PointerGeneratorDistributionModule, self).__init__()
        self.G = G
        self.unextended_embedding_layer = unextended_embedding_layer
        self.embedding_dim = self.unextended_embedding_layer.embedding_dim

    def forward(self, p_gen_list, vocab_distribution_list, attention_distribution_list, encoder_extended_vocab_input, oov_count):
        device = oov_count.device
        batch_size = oov_count.shape[0]

        vocab_distribution_list = [p_gen * dist for (p_gen, dist) in zip(p_gen_list, vocab_distribution_list)]
        attention_distribution_list = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gen_list, attention_distribution_list)]

        max_oov_count = oov_count.max()
        extended_vocab_size = self.unextended_embedding_layer.num_embeddings + max_oov_count
        extra_zeros = torch.zeros([batch_size, max_oov_count]).to(device)
        vocab_distribution_extended_list = [torch.cat([dist, extra_zeros], dim=1) for dist in vocab_distribution_list]


        attention_dists_projected = [torch.zeros([batch_size, extended_vocab_size]).to(device).scatter_add_(dim=1, index=encoder_extended_vocab_input, src=attn_dist) for attn_dist in attention_distribution_list]

        final_distribution_list = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_distribution_extended_list, attention_dists_projected)]

        return final_distribution_list

class PointerGeneratorNetwork(torch.nn.Module):

    def __init__(self, G):
        super(PointerGeneratorNetwork, self).__init__()
        self.G = G
        self.vocab_size = G.flags.vocab_size
        self.embedding_layer = torch.nn.Embedding(G.flags.vocab_size, G.flags.embedding_dim)
        self.encoder = VarLenEncoder(G, self.embedding_layer, G.flags.rnn_hidden_dim)
        self.encoder_state_reducer = StateReducer(G, self.encoder.rnn_hidden_dim)
        self.decoder = AttentionDecoder(G, self.embedding_layer, self.encoder.rnn_hidden_dim)
        self.output_projection = torch.nn.Linear(self.encoder.rnn_hidden_dim, self.vocab_size)
        self.final_distribution_module = PointerGeneratorDistributionModule(G, self.embedding_layer)
        self.criterion = PointerGeneratorLoss(G)

    def forward(self, *batch_input):
        encoder_original_vocab_input = batch_input[0]
        encoder_extended_vocab_input = batch_input[1]
        encoder_mask = batch_input[2]
        decoder_original_vocab_input = batch_input[3]
        decoder_extended_vocab_target = batch_input[4]
        decoder_mask = batch_input[5]
        oov_count = batch_input[6]

        device = encoder_original_vocab_input.device
        batch_size = encoder_original_vocab_input.shape[0]

        # in original code, encoder_output is used as "encoder_state"
        encoder_output, (encoder_h_t, encoder_c_t) = self.encoder(encoder_original_vocab_input, encoder_mask)
        reduced_encoder_state = self.encoder_state_reducer(encoder_h_t, encoder_c_t)
        # don't forget converage
        decoder_output_list, decoder_prev_step_state, attention_distribution_list, p_gen_list = self.decoder(decoder_original_vocab_input, encoder_output, reduced_encoder_state, encoder_mask)

        # out projection to obtain the vocabulary distribution
        vocab_score_list = [self.output_projection(output) for output in decoder_output_list]
        vocab_distribution_list = [F.softmax(s) for s in vocab_score_list]

        # calculate final dist
        final_distribution_list = self.final_distribution_module(p_gen_list, vocab_distribution_list, attention_distribution_list, encoder_extended_vocab_input, oov_count)

        # make loss
        loss = self.criterion(final_distribution_list, decoder_extended_vocab_target, decoder_mask)

        return loss

class PointerGeneratorLoss(torch.nn.Module):
    def __init__(self, G):
        super(PointerGeneratorLoss, self).__init__()
        self.G = G

    def forward(self, final_distribution_list, decoder_extended_vocab_target, decoder_mask):
        loss_per_step = []
        for step, dist in enumerate(final_distribution_list):
            step_target = decoder_extended_vocab_target[:, step]
            gold_probs = torch.gather(dist, 1, step_target.unsqueeze(-1))
            losses = -torch.log(gold_probs)
            loss_per_step.append(losses.squeeze())

        # add decoder mask and get final loss
        decoder_target_length = decoder_mask.sum(dim=-1)
        loss_list_per_step = [step_loss * decoder_mask[:, step] for step, step_loss in enumerate(loss_per_step)]
        loss_list_per_iter = sum(loss_list_per_step) / decoder_target_length
        return loss_list_per_iter.mean()


if __name__ == '__main__':
    main()
    print('all jobs done.')


