import json
import os
from tqdm import tqdm

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader


def train(loader, v_loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    #model = nn.DataParallel(model)
    with tqdm(range(opt["epochs"])) as epoch_bar:
        for epoch in epoch_bar:
            lr_scheduler.step()

            # If start self crit training
            if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
                sc_flag = True
                init_cider_scorer(opt["cached_tokens"])
            else:
                sc_flag = False

            train_loss, t_iter = forward(loader, model, crit, optimizer, sc_flag, rl_crit=rl_crit)

            epoch_bar.set_description('Epoch {}'.format(epoch))

            if epoch % opt["save_checkpoint_every"] == 0:
                val_loss, v_iter = forward(v_loader, model, crit, optimizer, sc_flag, train=False, rl_crit=rl_crit)
                model_path = os.path.join(opt["checkpoint_path"],
                                          'model_%d.pth' % (epoch))
                model_info_path = os.path.join(opt["checkpoint_path"],
                                               'model_score.txt')
                torch.save(model.state_dict(), model_path)
                #print("model saved to %s" % (model_path))
                with open(model_info_path, 'a') as f:
                    f.write("model_{}, loss: {:f}, val_loss: {:f}\n".format(epoch, train_loss/t_iter, val_loss/v_iter))


def forward(loader, model, crit, optimizer, sc_flag, train=True, rl_crit=None):
    iteration = 0
    total_loss = 0
    if train: model.train()
    else: model.eval()
    with tqdm(loader) as loader_bar:
        for data in loader_bar:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())

            if train:
                loader_bar.set_description('TRAIN')
                loss.backward()
                clip_grad_value_(model.parameters(), opt['grad_clip'])
                optimizer.step()
            else: loader_bar.set_description('VAL')
            split_loss = loss.item()
            total_loss += split_loss
            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
               loader_bar.set_postfix(loss=total_loss/iteration)
               # print("iter %d (epoch %d), train_loss = %.6f" %
               #       (iteration, epoch, train_loss))
            else:
               loader_bar.set_postfix(avg_reward='{:f}'.format(np.mean(reward[:,0])))
               # print("iter %d (epoch %d), avg_reward = %.6f" %
               #       (iteration, epoch, np.mean(reward[:, 0])))
    return total_loss, iteration

def main(opt):
    dataset, v_dataset = VideoDataset(opt, 'train'), VideoDataset(opt, 'val')
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
    v_loader = DataLoader(v_dataset, batch_size=opt['batch_size'], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(loader, v_loader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
