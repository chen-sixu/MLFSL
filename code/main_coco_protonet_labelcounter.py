"""Training script.
usage: main.py [options]
options:
    --learningrate=lr          Learning rate of inner loop [default: 0.001]
    --dataset=dataset           Dataset [default: coco]
    --datapath=datapath           Datapath [default: /home/csimon/research/data/ms-coco]
    --savepath=savepath           Savepath [default: ./save/coco_PN_resnet10_awal]
    --modeltype=model_type     Model Type [default: ConvNet]
    --batchsize=bs             Batch size [default: 1]
    --nway=nway                Number of classes [default: 10]
    --shot=shot                Number of shot [default: 5]
    --h, --help                Show help
"""
from docopt import docopt
from dataloader.episode_coco_set_k2 import CocoSet
from utils import pprint, accuracy_calc, process_label_coco_fewshot_definedlabel, Averager, acc_topk_definedlabel, create_mask, euclidean_metric, count_acc,count_acc_onehot
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from networks.cnn_encoder_relationet import CNNEncoder
from networks.convnet import ConvNet
from networks.backbone import ResNet10
from EpisodeSamplerMoreLabel import EpisodeSampler
import os.path as osp
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
from networks.selfsupervision import LabelCounter
from networks.cnn import CNN
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models
from dataloader.episode_nuswide_set_k2 import NusWideSet
import logging

#args = docopt(__doc__)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the desired logging level

# Create handlers
c_handler = logging.StreamHandler()       # Console handler
f_handler = logging.FileHandler('training.log')  # File handler

# Set levels for handlers if needed
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

def load_args(args):
    dd = {}
    dd['lr'] = 0.001#float(args['--learningrate'])
    dd['dataset'] = 'coco'#str(args['--dataset'])
    dd['datapath'] = str(args['--datapath'])
    dd['savepath'] = './save'#str(args['--savepath'])
    dd['modeltype'] = 'ConvNet'#str(args['--modeltype'])
    dd['batchsize'] = 1#int(args['--batchsize'])
    dd['shot'] = 1#int(args['--shot'])
    dd['nway'] = int(args['--nway'])

    return dd

# print(args)

# pprint(vars(args))

#args = load_args(args)

args={'lr':0.01, 'dataset':'COCO2017', 'datapath':'/root/autodl-tmp', 'savepath':'./save/imaterialist_mlpn',
      'modeltype':'ConvNet', 'batchsize':1, 'shot':1, 'nway':10}
logging.info(args)
all_label_size = {'coco':81}
total_label_size = args['nway']#all_label_size[args['dataset']]
batch_size = args['batchsize']
num_shot= args['shot']
query_size = total_label_size//2
feature_size = 1600
hidden_size= 8
total_traindata = 200
#sigma = 50
iter = 1000
greaterthan = 0.1

total_label_perepisode = total_label_size

aug=False
if args['modeltype'] == 'ResNet':
    aug=True

# trainset = CocoSet(args['datapath'], 'train','base', args, aug=aug)
# valset = CocoSet(args['datapath'], 'test','test', args,aug=aug)
if args['dataset']=='COCO2017' or args['dataset']=='COCO2014':
    trainset = CocoSet(args['datapath'], 'train','base', args, aug=aug)
    valset = CocoSet(args['datapath'], 'train','test', args,aug=aug)
elif args['dataset']=='NUS':
    trainset = NusWideSet(args['datapath'], 'train','base', args, aug=aug)
    valset = NusWideSet(args['datapath'], 'test','test', args,aug=aug)
train_sampler = EpisodeSampler(ids=trainset.ids, labels=trainset.label, label_ids=trainset.label_ids,
                               query_num=query_size, total_label_size=total_label_size, shot_size=num_shot,
                               n_batch=batch_size, iter=200)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)


val_sampler = EpisodeSampler(ids=valset.ids, labels=valset.label, label_ids=valset.label_ids,
                             query_num=query_size, total_label_size=total_label_size, shot_size=num_shot,
                             n_batch=batch_size, iter=iter)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

#testset = FlickrSet('test', args)

if args['modeltype'] == 'ConvNet':
    net = ConvNet().cuda('cuda:0')
    # net.load_state_dict(torch.load('/home/czh/code/clip-ML-FSL/save/best_model.pth'),strict=False)
    stepsize = 50
    #sigma = 50
    sigma = 50
    logging.info('convnet')
    label_counter = LabelCounter(feature_dim=1600).cuda('cuda:0')
else:
    net = ResNet10().cuda('cuda:0')
    stepsize = 120
    sigma = 30.
    label_counter = LabelCounter(feature_dim=512).cuda('cuda:0')

save_path = '-'.join([args['savepath'], args['modeltype'], 'cnn-rnn'])

#if args['model_type'] == 'ConvNet':
optimizer = torch.optim.Adam(list(net.parameters())+list(label_counter.parameters())+list(label_estimator.parameters()), lr=args['lr'])
# optimizer=torch.optim.SGD(list(net.parameters())+list(label_counter.parameters()), lr=args['lr'])

def kaiming_normal_init_net(net):
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

# Label estimator
class LE(nn.Module):
    def __init__(self, num_feature, num_classes, hidden_dim=128):
        super(LE, self).__init__()
        self.fe1 = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fe2 = nn.Linear(hidden_dim, hidden_dim)
        self.le1 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.le2 = nn.Linear(hidden_dim, hidden_dim)
        self.de1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_classes),
        )
        self.de2 = nn.Linear(num_classes, num_classes)

    def forward(self, x, y):
        x = self.fe1(x) + self.fe2(self.fe1(x))
        y = self.le1(y) + self.le2(self.le1(y))
        d = torch.cat([x, y], dim=-1)
        d = self.de1(d) + self.de2(self.de1(d))
        return d

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.5)
# elif args['model_type'] == 'ResNet':
#     optimizer = torch.optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, nesterov=True, weight_decay=0.0005)

def save_model(name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(dict(params=net.state_dict()), osp.join(save_path, name + '.pth'))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    net = net.cuda('cuda:0')

bceloss=torch.nn.BCEWithLogitsLoss()

maxs_train_F1 = .0
maxs_eval_F1 = 0.0
prototypes = torch.zeros(total_label_perepisode, feature_size).cuda('cuda:0')
np.seterr(all='raise')
best_epoch=0
best_map=0

label_estimator = LE(feature_size, total_label_size).cuda('cuda:0')
kaiming_normal_init_net(label_estimator)

def set_forward_loss(pred, le, x=None, y=None):
    assert y is not None
    loss_pred_cls = bceloss(pred, y)
    loss_pred_ld = nn.CrossEntropyLoss()(pred, torch.softmax(le.detach(), dim=1))
    loss_le_cls = loss_enhanced(le, pred, y)
    loss_le_spec = nn.CrossEntropyLoss()(le, torch.softmax(pred.detach(), dim=1))
        
    loss_pred = 0.001 * loss_pred_ld + 0.999 * loss_pred_cls
    loss_le = 0.001 * loss_le_spec + 0.999 * loss_le_cls
    return loss_le + loss_pred


def loss_enhanced(pred, teach, y):
    eps = 1e-7
    gamma1 = 0
    gamma2 = 1
    x_sigmoid = torch.sigmoid(pred)
    los_pos = y * torch.log(x_sigmoid.clamp(min=eps, max=1 - eps))
    los_neg = (1 - y) * torch.log((1 - x_sigmoid).clamp(min=eps, max=1 - eps))
    loss = los_pos + los_neg
    with torch.no_grad():
        teach_sigmoid = torch.sigmoid(teach)
        teach_pos = teach_sigmoid
        teach_neg = 1 - teach_sigmoid
        pt0 = teach_pos * y
        pt1 = teach_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma1 * y + gamma2 * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
    loss *= one_sided_w
    return -loss.mean()

for epoch in range(1, 500):

    cc = 0
    net.train()

    va_PR = Averager()
    va_RE = Averager()
    va_F1 = Averager()
    va_num = Averager()
    lossva = Averager()
    absolute_acc_count = Averager()
    loss = 0.0
    acc_all = []

    for jj, batch in enumerate(train_loader, 0):


        data, label_strs = [_ for _ in batch]
        #print(label_strs)
        must_label = train_sampler.get_querylabel(jj)
       # print(must_label)
        all_ori_label = train_sampler.get_total_ori_label(jj)
        label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                                                     must_exist_labels=must_label,
                                                                                                     total_label_episode=total_label_perepisode)


        label_queries = label_queries.cuda('cuda:0').sum(1) #sum over all onehot label for an image
        label_support = label_support.cuda('cuda:0')
        label_support_num = label_support.sum(-1).sum(1)
        label_queries_num = label_queries.sum(1)
        # print(label_queries.sum(0))
        #print(label_support_num)
        label_support_num_onehot = torch.zeros(label_support_num.shape[0], total_label_perepisode).cuda('cuda:0')
        label_support_num_onehot.fill_(0.)
        label_support_num_onehot.scatter_(1, label_support_num.data.unsqueeze(1), 1.)

        label_query_num_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode).cuda('cuda:0')
        label_query_num_onehot.fill_(0.)
        label_query_num_onehot.scatter_(1, label_queries_num.data.unsqueeze(1), 1.)


        data = data.cuda('cuda:0')

        bolong = []

        prototypes = []
        all_support_features = net(data[query_size:])
        #print(all_support_features)
        for key in range(total_label_perepisode):
            # if label_to_idx_support.get(key) == None:
            #     bolong.append(key)
            #     continue
            get_idx = np.array(label_to_idx_support[key])
            support_samples = net(data[get_idx])
            mean_samples = support_samples.mean(dim=0)
            prototypes.append(mean_samples)

        prototypes = torch.stack(prototypes)
        query_features = net(data[:query_size])
        raw_logits = euclidean_metric(query_features, prototypes)#relationnet(relation_pairs).view(-1, total_label_perepisode)
        temp_logits = -raw_logits
        le = label_estimator(query_features.detach(), label_queries)

        flem_loss = set_forward_loss(temp_logits, le, query_features, label_queries)

        ###### NEW BEGIN
        results, y_pred, num_label_ori_pred = label_counter(all_support_features, query_features, label_support_num)
        #gt_count_label_ori = torch.cat((label_queries_num, label_support_num), dim=0)
        gt_count_label = label_queries_num.unsqueeze(1) + label_support_num.unsqueeze(0)

        loss_count = F.cross_entropy(results.view(-1, results.shape[-1]), gt_count_label.view(-1).long())

        logits = F.softmax(raw_logits, dim=-1)
        #logits_clone = logits.clone()
        #label_queries_norm = label_queries.float()/torch.norm(label_queries.float(), p=1, dim=-1, keepdim=True)
        ############################ LOSSSS ##########################################################
        # loss = bceloss(logits.float()/0.05, label_queries.float()) + 0.01*loss_count
        loss = flem_loss + 0.01 * loss_count


        #INFERENCE
        y_pred_hard_constr = y_pred.clone()
        y_pred_hard_constr = y_pred_hard_constr.long().cuda('cuda:0')
        y_pred_hard_constr[y_pred_hard_constr < 0] = 0
        y_pred_hard_constr[y_pred_hard_constr > total_label_size - 1] =  total_label_size - 1
        y_pred_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode*2).cuda('cuda:0')
        y_pred_onehot.fill_(0.)
        y_pred_onehot.scatter_(1, y_pred_hard_constr.data.unsqueeze(1), 1.)


        absolute_acc = count_acc_onehot(logits, label_queries, topkpred=y_pred_onehot)
        acc = accuracy_calc(logits, label_queries.float(), label_size=total_label_size, greaterthan=greaterthan)
        acc_num = count_acc(y_pred_onehot, label_queries_num)#.float(), label_size=total_label_size, greaterthan=greaterthan)

        va_num.add(acc_num)
        absolute_acc_count.add(absolute_acc)
        ################## NEWWWW END

        va_PR.add(acc[0])
        va_RE.add(acc[1])
        va_F1.add(acc[2])
        lossva.add(loss)


        #std = np.std(acc_all) * 1.96 / np.sqrt(iter)


        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

    lr_scheduler.step()
    logging.info('[TRAIN] epoch {}'
          .format(epoch))

    if maxs_train_F1 < va_F1.item():
        maxs_train_F1 = va_F1.item()



    logging.info('[TRAIN] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f}, va_num={:.4f}, absolute_acc_count={:.4f}'
          .format(lossva.item(), va_PR.item(), va_RE.item(), va_F1.item(), maxs_train_F1, va_num.item(), absolute_acc_count.item()))


    save_model('last_epoch-'+ args['modeltype'] +'-coco-protonetsmax-'+str(args['nway'])+'-'+str(args['shot'])+'shot')

    if epoch % 5 != 0:
        continue


    va_val_F1 = Averager()
    va_val_PR = Averager()
    va_val_RE = Averager()
    lossva_val = Averager()
    va_num = Averager()
    absolute_acc_count = Averager()
    acc_all = []
    ap = 0.0

    for jj, batch in enumerate(val_loader, 0):
        net.eval()
        try:
            with torch.no_grad():


                data, label_strs = [_ for _ in batch]
                must_label = val_sampler.get_querylabel(jj)
                label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                                                             must_exist_labels=must_label,
                                                                                                             total_label_episode=total_label_perepisode)

                # label_queries = label_queries.cuda('cuda:0').sum(1) #sum over all onehot label for an image
                # label_support_val = label_support.cuda('cuda:0').sum(1)
                label_queries = label_queries.cuda('cuda:0').sum(1) #sum over all onehot label for an image
                label_support = label_support.cuda('cuda:0')
                label_support_num = label_support.sum(-1).sum(1)
                label_queries_num = label_queries.sum(1)

                label_support_num_onehot = torch.zeros(label_support_num.shape[0], total_label_perepisode).cuda('cuda:0')
                label_support_num_onehot.fill_(0.)
                label_support_num_onehot.scatter_(1, label_support_num.data.unsqueeze(1), 1.)

                label_query_num_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode).cuda('cuda:0')
                label_query_num_onehot.fill_(0.)
                label_query_num_onehot.scatter_(1, label_queries_num.data.unsqueeze(1), 1.)


                data = data.cuda('cuda:0')


                bolong = []
                prototypes = []#prototypes * 0.
                all_support_features = net(data[query_size:])
                for key in range(total_label_perepisode):
                    # if label_to_idx_support.get(key) == None:
                    #     bolong.append(key)
                    #     continue
                    get_idx = np.array(label_to_idx_support[key])
                    support_samples = net(data[get_idx])
                    mean_samples = support_samples.mean(dim=0)
                    prototypes.append(mean_samples)

                prototypes = torch.stack(prototypes)
                query_features = net(data[:query_size])
                logits = euclidean_metric(query_features, prototypes)#relationnet(relation_pairs).view(-1, total_label_perepisode)


               # label_queries_norm = label_queries.float()/torch.norm(label_queries.float(), p=1, dim=-1, keepdim=True)


                results, y_pred, _ = label_counter(all_support_features, query_features, label_support_num)

                logits = F.softmax(logits, dim=-1)

                loss = 0.0#sparsemaxLoss(logits, zs_sparse, label_queries_norm.float(), logits_spars, taus, is_gt)



                y_pred_hard_constr = y_pred.clone()
                y_pred_hard_constr = y_pred_hard_constr.long().cuda('cuda:0')
                y_pred_hard_constr[y_pred_hard_constr < 0] = 0
                y_pred_hard_constr[y_pred_hard_constr > total_label_size - 1] =  total_label_size - 1
                y_pred_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode*2).cuda('cuda:0')
                y_pred_onehot.fill_(0.)
                y_pred_onehot.scatter_(1, y_pred_hard_constr.data.unsqueeze(1), 1.)

                absolute_acc = count_acc_onehot(logits, label_queries, topkpred=y_pred_onehot)
                acc = accuracy_calc(logits, label_queries.float(), label_size=total_label_size, greaterthan=greaterthan)
                acc_num = count_acc(y_pred_onehot, label_queries_num)
                #acc_all.append(acc[2])


                #acc = accuracy_calc(logits, label_queries.float(), label_size=total_label_size, greaterthan=greaterthan, topk=3)
                va_val_PR.add(acc[0])
                va_val_RE.add(acc[1])
                va_val_F1.add(acc[2])
                lossva_val.add(loss)
                absolute_acc_count.add(absolute_acc)
                va_num.add(acc_num)

                #absolute_acc = count_acc_onehot()

                logits_np = logits.cpu().numpy()
                label_np = label_queries.cpu().numpy()
                ap_temp = 0.0
                for bb in range(logits_np.shape[0]):
                    ap_temp += average_precision_score(label_np[bb], logits_np[bb])
                ap += ap_temp/logits_np.shape[0]
                acc_all.append(ap_temp/logits_np.shape[0])


                # print('epoch {}, eval {}/{}, loss={:.4f} acc={:.4f} avg={:.4f} '
                #       .format(epoch, jj, total_traindata//batch_size, loss.item(), acc, va_val.item()))




        except:
            print("FAIL")

    std = np.std(acc_all) * 1.96 / np.sqrt(iter)
    logging.info('Final {}:     F1={:.2f}({:.2f})'.format(epoch, np.mean(acc_all) * 100, std * 100))

    if maxs_eval_F1 < va_val_F1.item():
        maxs_eval_F1 = va_val_F1.item()
        save_model('best_epoch-'+ args['modeltype'] +'-coco-protonetsmax-'+str(args['nway'])+'-'+str(args['shot'])+'shot')
    if best_map < ap/iter:
        best_map = ap/iter
        best_epoch = epoch
    logging.info('[EVAL] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f} map={:.4f}, va-num={:.4f}, absolute_acc_count={:.4f}'
          .format(lossva.item(), va_val_PR.item(), va_val_RE.item(), va_val_F1.item(), maxs_eval_F1, ap/iter, va_num.item(), absolute_acc_count.item()))
    logging.info('best map : ' + str(best_map) + ' at epoch ' + str(best_epoch))
