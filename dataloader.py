import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

         # load the json file which contains information about the dataset
        self.captions = json.load(open(opt['caption_json']))
        info = json.load(open(opt['info_json']))
        self.splits = info['videos']
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt['feats_dir']
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        self.index_map = json.load(open(opt['index_clip_mapping']))
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt['max_len']
        print('max sequence length in data is', self.max_len)


    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
#        if self.with_c3d == 1:
#            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, npy_name))
#            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
#            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)

        global_clip_id = self.global_clip_id(ix)
        npy_name = self.npy_name(global_clip_id)
        fc_feat = self.fc_feats(npy_name)
        captions = self.captions[global_clip_id]['final_captions']
        gts = self.gts(global_clip_id, captions)
        label = gts[self.cap_ix(captions)]
        non_zero = (label == 0).nonzero()
        mask = np.zeros(self.max_len)
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = global_clip_id
        return data


    def global_clip_id(self, ix):
        return str(self.splits[self.mode][ix])


    def npy_name(self, global_clip_id):
        clip_name = self.index_map['clips'][global_clip_id]
        return '{}.npy'.format(clip_name)


    def fc_feats(self, npy_name):
        fc_feat = []
        for dir in self.feats_dir:
            feat = np.load(os.path.join(dir, npy_name))

            if len(feat) == 1:
                feat = np.repeat(feat, 40, axis=0)

            fc_feat.append(feat)
        fc_feat = np.concatenate(fc_feat, axis=1)
        return fc_feat

#            fc_feat.append(np.load(os.path.join(dir, npy_name)).flatten())
#        fc_feat = np.hstack(fc_feat)
#        return np.expand_dims(fc_feat, 0)

#        return np.concatenate([np.load(os.path.join(d, npy_name))
#                               for d in self.feats_dir], axis=1)


    def gts(self, global_clip_id, captions):
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]
        return gts


    def cap_ix(self, captions):
#        return random.randint(0, len(captions) - 1)
        return 0 # MPII dataset only has 1 sentence per clip


    def __len__(self):
        return len(self.splits[self.mode])

