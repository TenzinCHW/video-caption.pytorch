import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MovieLoader():
    def __init__(self, opt, mode, shuffle=True):
        self.mode = mode  # to load train/val/test data
        self.shuffle = shuffle

         # load the json file which contains information about the dataset
        self.captions = json.load(open(opt['caption_json']))
        info = json.load(open(opt['info_json']))
        self.splits = info['movies']
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['movies']
        print('number of train movies: ', len(self.splits['train']))
        print('number of val movies: ', len(self.splits['val']))
        print('number of test movies: ', len(self.splits['test']))

        self.feats_dir = opt['feats_dir']
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        self.index_map = json.load(open(opt['index_clip_mapping']))
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt['max_len']
        print('max sequence length in data is', self.max_len)
        self.batch_size = opt['batch_size']

        self.opt_ref = {'caption' : self.captions,
                        'ix_to_word' : self.ix_to_word,
                        'word_to_ix' : self.word_to_ix,
                        'feats_dir' : self.feats_dir,
                        'index_map' : self.index_map,
                        'max_len' : self.max_len,
                        'c3d_feats_dir' : self.c3d_feats_dir,
                        'with_c3d' : self.with_c3d,
                        'batch_size' : self.batch_size}

        # Control split of data to load
        movie_ids = [self.splits[self.mode][ix]
                     for ix in range(len(self))]
        movie_sets = [VideoDataset(self.opt_ref, movie_id)
                      for movie_id in movie_ids]
        self.loaders = [DataLoader(movie_set, batch_size=1,
                             shuffle=True)
                             for movie_set in movie_sets]

#        self.loaders = [DataLoader(movie_set, batch_size=self.batch_size,
#                             shuffle=False)
#                             for movie_set in movie_sets]

    def __iter__(self):
        self.n = 0
        if self.shuffle:
            random.shuffle(self.loaders)
        return self

    def __next__(self):
        if self.n < len(self):
            next_item = self.loaders[self.n]
            self.n += 1
            return next_item
        else:
            raise StopIteration

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def __len__(self):
        return len(self.splits[self.mode])


class VideoDataset(Dataset):

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, movie_id):
        super(VideoDataset, self).__init__()
        self.movie_id = movie_id
        self.feats_dir =  opt['feats_dir']
        self.captions = opt['caption']
        self.ix_to_word = opt['ix_to_word']
        self.word_to_ix = opt['word_to_ix']
        self.index_map = opt['index_map']
        self.max_len = opt['max_len']
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        self.batch_size = opt['batch_size']


    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        global_clip_ids = [self.global_clip_id(i) for i in range(ix, ix+self.batch_size)]
        npy_names = [self.npy_name(global_clip_id) for global_clip_id in global_clip_ids]
        fc_feats = [self.fc_feats(npy_name) for npy_name in npy_names]

#        if self.with_c3d == 1:
#            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, npy_name))
#            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
#            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)

        captions = [self.captions[global_clip_id]['final_captions'] for global_clip_id in global_clip_ids]
        gts = [self.gts(global_clip_ids[i], caps) for i, caps in enumerate(captions)]

        # random select a caption for this video
        labels = [gts[self.cap_ix(caps)] for caps in captions]
        non_zeros = [(label == 0).nonzero() for label in labels]
        masks = [np.zeros(self.max_len) for nz in non_zeros]
        for nz, mask in zip(non_zeros, masks):
            mask[:int(nz[0][0]) + 1] = 1

        fc_feats = np.concatenate(fc_feats)
        labels = np.squeeze(np.concatenate(labels, axis=1))
        masks = np.concatenate(masks)
        gts = np.concatenate(gts)

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feats).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(labels).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(masks).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = global_clip_ids
        return data


    def global_clip_id(self, ix):
        return str(self.index_map['movies'][self.movie_id][ix])


    def npy_name(self, global_clip_id):
        clip_name = self.index_map['clips'][global_clip_id]
        return '{}.npy'.format(clip_name)


    def fc_feats(self, npy_name):
#        fc_feat = []
#        for dir in self.feats_dir:
#            fc_feat.append(np.load(os.path.join(dir, npy_name)).flatten())
#        fc_feat = np.hstack(fc_feat)
#        fc_feat = np.expand_dims(fc_feat, 0)
        return np.concatenate([np.load(os.path.join(d, npy_name))
                               for d in self.feats_dir], axis=1)


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
        #return random.randint(0, len(captions) - 1)
        return 0 # MPII dataset only has 1 sentence per clip


    def __len__(self):
        return len(self.index_map['movies'][self.movie_id]) - self.batch_size
#        return len(self.index_map['movies'][self.movie_id])

