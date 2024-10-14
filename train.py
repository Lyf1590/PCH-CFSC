import math
import torch.utils.data as data
import dset
import metric
from metric import *
from layers import ImageMlp, TextMlp, MDE
from metric import affinity_tag_multi, knn1
from models import PGCH, FuseTransEncoder, Encoder
import settings
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp


class Session:
    def __init__(self):
        self.logger = settings.logger
        self.alpha_train = settings.alpha_train
        self.beta_train = settings.beta_train
        self.alpha_query = settings.alpha_query
        self.beta_query = settings.beta_query
        self.COM_PART = False
        self.logger.info('------------------------------------------------------------------------------------')
        self.logger.info('The Dataset is %s! The epoch_num is %d! The number of hash codes is %d bits!' %
                         (settings.DATASET_NAME, settings.NUM_EPOCH, settings.CODE_LEN))
        self.logger.info('------------------------------------------------------------------------------------')

        self.logger.info('------------------------------------------------------------------------------------')

        # 分别存放训练、测试、数据库数据集
        complete_data, train_missed_data, query_missed_data = dset.load_data(self.alpha_train, self.beta_train, self.alpha_query, self.beta_query)

        if train_missed_data['dual_label'].shape[0] == complete_data['L_tr'].shape[0]:
            self.COM_PART = True
            self.logger.info('**********No missing data samples!**********')
        else:
            self.logger.info('**********There are missing data samples!**********')
            self.logger.info('Dual Complete ratio is %.1f!  Only image ratio is %.2f!  Only text ratio is %.2f!' % (self.alpha_train, (1-self.alpha_train)*self.beta_train, (1-self.alpha_train)*self.beta_train))

        self.train_data = [complete_data['I_tr'], complete_data['T_tr']]
        self.train_labels = complete_data['L_tr'].numpy()
        self.query_data = [complete_data['I_te'], complete_data['T_te']]
        self.query_labels = complete_data['L_te'].numpy()
        self.retrieval_data = [complete_data['I_db'], complete_data['T_db']]
        self.retrieval_labels = complete_data['L_db'].numpy()

        # 缺失数据
        self.train_dual_data = [train_missed_data['dual_img'], train_missed_data['dual_txt']]
        self.train_dual_labels = train_missed_data['dual_label']
        self.train_only_imgs = train_missed_data['o_img']
        self.train_only_imgs_labels = train_missed_data['o_img_label']
        self.train_only_txts = train_missed_data['o_txt']
        self.train_only_txts_labels = train_missed_data['o_txt_label']

        self.query_dual_data = [query_missed_data['dual_img'], query_missed_data['dual_txt']]
        self.query_dual_labels = query_missed_data['dual_label']
        self.query_only_imgs = query_missed_data['o_img']
        self.query_only_imgs_labels = query_missed_data['o_img_label']
        self.query_only_txts = query_missed_data['o_txt']
        self.query_only_txts_labels = query_missed_data['o_txt_label']

        self.num_classes = self.train_labels.shape[1]
        self.train_nums = self.train_labels.shape[0]
        self.train_dual_nums = self.train_dual_data[0].size(0)
        self.train_only_imgs_nums = self.train_only_imgs.size(0)
        self.train_only_imgs_dims = self.train_only_imgs.size(1)
        self.train_only_txts_nums = self.train_only_txts.size(0)
        self.train_only_txts_dims = self.train_only_txts.size(1)
        # 图像数据的维度
        self.img_dim = self.train_data[0].size(1)
        # 文本数据的维度
        self.txt_dim = self.train_data[1].size(1)
        assert self.train_nums == (self.train_dual_nums + self.train_only_imgs_nums + self.train_only_txts_nums)

        self.batch_dual_size = math.ceil(settings.BATCH_SIZE * self.alpha_train)
        self.batch_img_size = math.floor((settings.BATCH_SIZE - self.batch_dual_size) * self.beta_train)
        self.batch_txt_size = settings.BATCH_SIZE - self.batch_dual_size - self.batch_img_size
        self.batch_count = int(math.ceil(self.train_nums / settings.BATCH_SIZE))

        # 锚点选择
        self.anchor_nums = settings.ANCHOR_NUMS
        if self.anchor_nums > self.train_dual_nums:
            self.logger.critical('The anchor number is large than the number of dual samples.')

        # 加载数据集
        self.train_loader = data.DataLoader(dset.MY_DATASET(self.train_dual_data[0], self.train_dual_data[1], self.train_dual_labels),
                                                   batch_size=settings.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=True)  # drop_last：丢弃最后一个不完整批次

        self.test_loader = data.DataLoader(dset.MY_DATASET(self.query_data[0], self.query_data[1], self.query_labels),
                                                    batch_size=settings.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=settings.NUM_WORKERS)

        self.database_loader = data.DataLoader(dset.MY_DATASET(self.retrieval_data[0], self.retrieval_data[1], self.retrieval_labels),
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)

        # 创建各个模型实例
        self.ImgDe = MDE(hidden_dim=[512, 2048, 1024, settings.CODE_LEN])
        self.TxtDe = MDE(hidden_dim=[512, 2048, 1024, settings.CODE_LEN])
        self.gcn = PGCH(bits=settings.CODE_LEN, classes=self.num_classes)  # feature_t=settings.feature_t
        self.opt_G = torch.optim.Adam(self.gcn.parameters(), lr=0.01)
        self.opt_D = torch.optim.Adam([
            {"params": self.ImgDe.parameters(), "lr": 1e-4},
            {"params": self.TxtDe.parameters(), "lr": 1e-4}])

        self.feat_lens = 512
        num_layers, self.token_size, nhead = 2, 1024, 4
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).cuda()
        paramsFuse_to_update = list(self.FuseTrans.parameters()) 
        self.optimizer_FuseTrans = torch.optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))

        self.ImageMlp = ImageMlp(self.feat_lens, settings.CODE_LEN).cuda()
        self.TextMlp = TextMlp(self.feat_lens, settings.CODE_LEN).cuda()

        self.img_enc = Encoder(input_dim=self.txt_dim, output_dim=self.img_dim, dim_feedforward=1024, k=self.anchor_nums).cuda()
        self.txt_enc = Encoder(input_dim=self.img_dim, output_dim=self.txt_dim, dim_feedforward=1024, k=self.anchor_nums).cuda()

        self.TEs_optimizer = torch.optim.Adam(
            [{"params": self.img_enc.parameters(), "lr": 1e-3},
             {"params": self.txt_enc.parameters(), "lr": 1e-3}])

    def com_train(self, epoch):
        self.gcn.cuda().train()
        self.FuseTrans.cuda().train()
        epoch_loss = 0
        loss_fn = torch.nn.MSELoss()
        loss_adv = torch.nn.BCELoss()

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            img = Variable(torch.FloatTensor(img.numpy()).cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            labels = Variable(torch.FloatTensor(labels.numpy()).cuda())
            _, aff_norm, aff_label = affinity_tag_multi(labels.cpu().numpy(), labels.cpu().numpy())
            aff_label = Variable(torch.FloatTensor(aff_label).cuda())

            self.opt_G.zero_grad()
            self.optimizer_FuseTrans.zero_grad()
            img_common, txt_common, hash1, hash2, pred1, pred2, D1, D2 = self.gcn(img, txt)

            temp_tokens = torch.cat((img_common, txt_common), dim=1)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            loss_con1 = metric.SimCLR(img_embedding, text_embedding)
            img_embedding = self.ImageMlp(img_embedding)
            text_embedding = self.TextMlp(text_embedding)
            loss_con2 = metric.SimCLR(img_embedding, text_embedding)
            loss_con = (loss_con1 + loss_con2) * settings.Alpha  # 对比损失

            recon_loss1 = F.cross_entropy(pred1, torch.argmax(labels, dim=1))
            recon_loss2 = F.cross_entropy(pred2, torch.argmax(labels, dim=1))
            recon_loss = recon_loss1 + recon_loss2
            clf_loss1 = loss_fn(torch.sigmoid(pred1), labels)
            clf_loss2 = loss_fn(torch.sigmoid(pred2), labels)
            clf_loss = clf_loss1 + clf_loss2
            pred_loss = (recon_loss + clf_loss) * settings.Lambda  # 预测损失


            Binary1 = torch.sign(hash1)
            Binary2 = torch.sign(hash2)
            loss7 = loss_fn(hash1, Binary1)
            loss8 = loss_fn(hash2, Binary2)
            sig_loss = (loss7 + loss8) * settings.Gamma  # 二值损失


            H1_norm = F.normalize(hash1)
            H2_norm = F.normalize(hash2)
            similarity_loss = settings.Eta * (loss_fn(H1_norm.mm(H1_norm.t()), aff_label) + loss_fn(
                H2_norm.mm(H2_norm.t()), aff_label)) # 相似性损失

            loss1 = loss_adv(D1, Variable(torch.zeros(D1.shape[0], 1)).cuda())
            loss2 = loss_adv(D2, Variable(torch.ones(D2.shape[0], 1)).cuda())
            lossd = loss1 + loss2  # 辨别哈希码损失1
            hash_loss = loss_fn(hash1, hash2)  # 辨别损失2
            dis_loss = (lossd + hash_loss) * settings.Beta

            loss = loss_con + pred_loss + sig_loss + similarity_loss + dis_loss


            loss.backward()
            self.opt_G.step()
            self.optimizer_FuseTrans.step()

            if (idx + 1) % (self.train_nums // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH, idx + 1, self.train_nums // settings.BATCH_SIZE,
                        loss.item()))
            epoch_loss = loss.item()

        return epoch_loss

    def partial_train(self):
        self.gcn.cuda().train()
        self.FuseTrans.cuda().train()
        self.ImgDe.cuda().train()
        self.TxtDe.cuda().train()
        self.img_enc.cuda().train()
        self.txt_enc.cuda().train()

        Losses = []
        best_it = best_ti = 0
        dual_idx = np.arange(self.train_dual_nums)
        oimg_idx = np.arange(self.train_only_imgs_nums)
        otxt_idx = np.arange(self.train_only_txts_nums)
        for epoch in range(settings.NUM_EPOCH):
            np.random.shuffle(dual_idx)
            np.random.shuffle(oimg_idx)
            np.random.shuffle(otxt_idx)
            epoch_loss = 0
            for batch_idx in range(self.batch_count):
                small_idx_dual = dual_idx[batch_idx * self.batch_dual_size: (batch_idx + 1) * self.batch_dual_size]
                small_idx_img = oimg_idx[batch_idx * self.batch_img_size: (batch_idx + 1) * self.batch_img_size]
                small_idx_txt = otxt_idx[batch_idx * self.batch_txt_size: (batch_idx + 1) * self.batch_txt_size]

                train_dual_img = torch.FloatTensor(self.train_dual_data[0][small_idx_dual, :]).cuda()
                train_dual_txt = torch.FloatTensor(self.train_dual_data[1][small_idx_dual, :]).cuda()
                train_dual_labels = self.train_dual_labels[small_idx_dual, :].cuda()

                train_only_img = torch.FloatTensor(self.train_only_imgs[small_idx_img, :]).cuda()
                train_only_img_labels = self.train_only_imgs_labels[small_idx_img, :].cuda()

                train_only_txt = torch.FloatTensor(self.train_only_txts[small_idx_txt, :]).cuda()
                train_only_txt_labels = self.train_only_txts_labels[small_idx_txt, :].cuda()

                loss = self.partial_step(train_dual_img, train_dual_txt, train_dual_labels, train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels)

                if (batch_idx + 1) % (self.train_nums // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                    self.logger.info('Epoch [%d/%d], Iter [%d/%d] Total Loss: %.4f' % (epoch + 1, settings.NUM_EPOCH, batch_idx + 1,
                                                                                       self.train_nums // settings.BATCH_SIZE, loss.item()))
                epoch_loss = loss.item()

            # return epoch_loss
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = self.eval()
                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    print('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))
                    self.logger.info('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))
                    self.save_checkpoints('checkpoint')
                self.logger.info('Final  Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))

    def partial_step(self, train_dual_img, train_dual_txt, train_dual_labels, train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels):
        loss_fn = torch.nn.MSELoss()
        loss_adv = torch.nn.BCELoss()
        self.opt_G.zero_grad()
        self.opt_D.zero_grad()
        self.optimizer_FuseTrans.zero_grad()
        self.TEs_optimizer.zero_grad()

        img_forward = torch.cat([train_dual_img, train_only_img])
        txt_forward = torch.cat([train_dual_txt, train_only_txt])
        img_labels = torch.cat([train_dual_labels, train_only_img_labels])
        txt_labels = torch.cat([train_dual_labels, train_only_txt_labels])

        txt_anchor = knn1(self.anchor_nums, txt_forward)
        img_anchor = knn1(self.anchor_nums, img_forward)

        img_neighbour = self.img_enc(train_only_txt, txt_anchor).cuda()
        txt_neighbour = self.txt_enc(train_only_img, img_anchor).cuda()

        img_all = torch.cat((train_dual_img, train_only_img, train_only_txt), dim=0)
        txt_all = torch.cat((train_dual_txt, train_only_img, train_only_txt), dim=0)
        labels_all = torch.cat([train_dual_labels, train_only_img_labels, train_only_txt_labels])
        _, _, aff_img_lable = affinity_tag_multi(img_labels.cpu().numpy(), img_labels.cpu().numpy())
        _, _, aff_txt_lable = affinity_tag_multi(txt_labels.cpu().numpy(), txt_labels.cpu().numpy())
        _, _, aff_label = affinity_tag_multi(labels_all.cpu().numpy(), labels_all.cpu().numpy())
        aff_img_lable = Variable(torch.FloatTensor(aff_img_lable).cuda())
        aff_txt_lable = Variable(torch.FloatTensor(aff_txt_lable).cuda())
        aff_label = Variable(torch.FloatTensor(aff_label).cuda())


        img_common_, txt_common_, hash1_, hash2_, pred1_, pred2_, D1_, D2_ = self.gcn(img_all, txt_all)

        loss_rec1 = loss_fn(train_only_txt, img_neighbour)
        loss_rec2 = loss_fn(train_only_img, txt_neighbour)
        gen_loss = (loss_rec1 + loss_rec2) * settings.Mu  # 生成损失gen_loss
        # LCMR
        # gen_loss = 0

        # 截断批量维度使其一致
        min_batch_size = min(img_common_.shape[0], txt_common_.shape[0])
        img_common_ = img_common_[:min_batch_size]
        txt_common_ = txt_common_[:min_batch_size]
        temp_tokens_ = torch.cat((img_common_, txt_common_), dim=1)
        temp_tokens_ = temp_tokens_.unsqueeze(0)
        img_embedding_, text_embedding_ = self.FuseTrans(temp_tokens_)
        loss_con1_ = metric.SimCLR(img_embedding_, text_embedding_)
        img_embedding_ = self.ImageMlp(img_embedding_)
        text_embedding_ = self.TextMlp(text_embedding_)
        loss_con2_ = metric.SimCLR(img_embedding_, text_embedding_)
        cont_loss = (loss_con1_ + loss_con2_) * settings.Alpha  # 对比损失
        # L SCL
        # cont_loss = 0

        recon_loss1_ = F.cross_entropy(pred1_, torch.argmax(labels_all, dim=1))  # 预测损失
        recon_loss2_ = F.cross_entropy(pred2_, torch.argmax(labels_all, dim=1))
        recon_loss_ = recon_loss1_ + recon_loss2_

        clf_loss1 = loss_fn(torch.sigmoid(pred1_), labels_all)
        clf_loss2 = loss_fn(torch.sigmoid(pred2_), labels_all)
        clf_loss = clf_loss1 + clf_loss2
        pred_loss = (recon_loss_ + clf_loss) * settings.Lambda  # 预测损失=1000
        # L CPL
        #  pred_loss = 0

        H1_norm = F.normalize(hash1_)
        H2_norm = F.normalize(hash2_)  # 相似性损失 Eta=10
        similarity_loss = settings.Eta * (loss_fn(H1_norm.mm(H1_norm.t()), aff_label) + loss_fn(H2_norm.mm(H2_norm.t()), aff_label))
        # L SIM
        # similarity_loss = 0

        # 截断批量维度使其一致
        min_batch_size = min(hash1_.shape[0], hash2_.shape[0])
        hash1_ = hash1_[:min_batch_size]
        hash2_ = hash2_[:min_batch_size]
        hash_loss_ = loss_fn(hash1_, hash2_)

        loss1 = loss_adv(D1_, Variable(torch.zeros(D1_.shape[0], 1)).cuda())
        loss2 = loss_adv(D2_, Variable(torch.ones(D2_.shape[0], 1)).cuda())
        lossd = loss1 + loss2
        dis_loss_ = (lossd + hash_loss_) * settings.Beta  # 辨别哈希损失 0.001
        # L DIS
        # dis_loss_ = 0


        Binary1_ = torch.sign(hash1_)
        Binary2_ = torch.sign(hash2_)
        loss_bin1 = loss_fn(hash1_, Binary1_)
        loss_bin2 = loss_fn(hash2_, Binary2_)
        sig_loss_ = (loss_bin1 + loss_bin2) * settings.Gamma  # 二值损失 Gamma2:10
        # L BIN
        # sig_loss_ = 0

        loss = gen_loss + cont_loss + pred_loss + similarity_loss + sig_loss_ + dis_loss_
        Loss = loss
        Loss.backward()
        self.opt_G.step()
        self.opt_D.step()
        self.optimizer_FuseTrans.step()
        self.TEs_optimizer.step()
        return Loss

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        # self.logger.info('--------------------Precision@top-R Curve-------------------')
        self.gcn.cuda().eval()
        self.FuseTrans.cuda().eval()
        self.ImgDe.cuda().eval()
        self.TxtDe.cuda().eval()
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.gcn)
        # 1. mAP
        MAP_I2T_50 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I_50 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        return MAP_I2T_50, MAP_T2I_50

        # # 2.topN
        # P_topK_I2T = p_topK2(qB=qu_BI, rB=re_BT, query_label=qu_L, retrieval_label=re_L, K=settings.topN_Number)
        # P_topK_T2I = p_topK2(qB=qu_BT, rB=re_BI, query_label=qu_L, retrieval_label=re_L, K=settings.topN_Number)
        #
        # #3.P-R 图
        # self.logger.info('=====================================================================================')
        # p_i2t = pr_curve(qB=qu_BI, rB=re_BT, qL=qu_L, rL=re_L)
        # p_t2i = pr_curve(qB=qu_BT, rB=re_BI, qL=qu_L, rL=re_L)
        # return P_topK_I2T, P_topK_T2I, p_i2t, p_t2i

    def save_checkpoints(self, step=None):
        if step is not None:
            torch.save(self.gcn.state_dict(), os.path.join('checkpoint', 'checkpoint_GCN.pth'))
            torch.save(self.FuseTrans.state_dict(), os.path.join('checkpoint', 'checkpoint_FuseTrans.pth'))

    def load_checkpoints(self):
        filename = ['checkpoint_GCN.pth', 'checkpoint_FuseTrans.pth']
        ckp_path = osp.join('checkpoint', filename[0])
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError




