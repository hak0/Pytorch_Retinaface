import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        # predictions为一个tuple，这里把各部分的prediction拿出来，每个都是一个tensor
        # loc_data的shape为(batchsize32, anchor数16800, bbox位置预测4)
        # conf_data的shape为(batchsize32, anchor数16800, 是否为人脸2)
        # landm_data的shape为(batchsize32, anchor数16800, 特征点坐标10)
        loc_data, conf_data, landm_data = predictions
        # priors是训练之前就已经生成的先验框的坐标，也就是anchors
        priors = priors
        # num即batch num，一个batch里有多少个样本
        num = loc_data.size(0)
        # num_priors为anchor总数，为16800
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        # 把label的各部分切开来，truth存储bbox位置，labels存储是否为人脸，landms存储特征点坐标
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            # defaults为anchors，shape为(16800, 4)
            # 这里这个match函数把ground truth的框匹配到最接近的anchor上
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        # pos1为预测中confidence>0的样本的mask
        pos1 = conf_t > zeros
        # num_pos_landm为本batch中每个sample对应的conf>0的anchors的数目
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        # N1为本batch中所有conf>0的anchors总数，也就是num_pos_landm的sum
        N1 = max(num_pos_landm.data.sum().float(), 1)
        # pos_idx1为之前的pos1扩展到landm_data形状的mask，接下来的操作都要在该mask上进行
        # 也就是说，loss的计算仅对confidence>0的landmark生效
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # 这里利用前面的mask筛选出conf预测值>0的样本，然后用view挑出来，形状还是n行10列，n为挑出的样本数
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # 和之前的pos_idx1类似，把pos1扩展到loc_data形状的mask
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # 这里batch_conf就是把本batch所有anchors的confidence全都排列到一起了
        # 变成了一个很多行，两列的tensor，第一列是“是人脸”的置信度分数，第二列是“是背景”的置信度分数
        batch_conf = conf_data.view(-1, self.num_classes)
        # 然后计算置信度损失。
        # 第一个是把所有的batch_conf给log_sum_exp，
        # 第二个是根据groundtruth的标签（是人脸还是背景）来选取每行对应的那个confidence，
        # 并把这个confidence给reshape成一列
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        # 现在loss_c是每一个anchor的负样本的loss
        # 根据负样本loss排序，并选出那些loss特别大的负样本（即难样本）
        # loss_idx的shape为(32, 16800)。
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # num_pos和num_neg为本batch每张图的所有anchors中挑出的正负样本的个数
        # neg与之前的pos类似，是挑出的负样本的mask
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # 只是OHEM了一下，把负样本挑了一部分置信度特别低的出来
        # 然后算cross-entropy
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
