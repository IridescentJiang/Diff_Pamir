"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import time
import sys
import torch.nn.functional as F
from skimage import measure
import scipy.io as sio
import datetime
import glob
import logging
import math
from tqdm import tqdm

from util.base_trainer import BaseTrainer
from dataloader.dataloader import TrainingImgDataset
from network.arch import PamirNet
from neural_voxelization_layer.smpl_model import TetraSMPL
from neural_voxelization_layer.voxelize import Voxelization
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor, SMPL
from graph_cmr.utils import Mesh
from graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import util.obj_io as obj_io
import util.util as util
import constant as const
from util.data_loader import CheckpointDataLoader
from diffusion.MLPdiffusion import MLPDiffusion


class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def init_fn(self):
        super(BaseTrainer, self).__init__()
        # dataset
        self.train_ds = TrainingImgDataset(
            self.options.dataset_dir, img_h=const.img_res, img_w=const.img_res,
            training=True, testing_res=256,
            view_num_per_item=self.options.view_num_per_item,
            point_num=self.options.point_num,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')

        # GraphCMR components
        self.img_norm = ImgNormalizerForResnet().to(self.device)
        self.graph_mesh = Mesh()
        self.graph_cnn = GraphCNN(self.graph_mesh.adjmat, self.graph_mesh.ref_vertices.t(),
                                  const.cmr_num_layers, const.cmr_num_channels).to(self.device)
        self.smpl_param_regressor = SMPLParamRegressor().to(self.device)

        # neural voxelization components
        self.smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        self.tet_smpl = TetraSMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=self.options.batch_size).to(self.device)

        # pamir_net
        self.pamir_net = PamirNet().to(self.device)

        # diffusion
        self.num_steps = 100 # 即T,对于步骤，一开始可以由beta, 分布的均值和标准差来共同确定
        self.diffusion_model = MLPDiffusion(self.num_steps).to(self.device)

        # optimizers
        self.optm_pamir_net = torch.optim.RMSprop(
            params=list(self.pamir_net.parameters()), lr=float(self.options.lr)
        )
        self.optm_diffusion_model = torch.optim.RMSprop(
            params=list(self.diffusion_model.parameters()), lr=float(self.options.lr)
        )

        # loses
        self.criterion_geo = nn.MSELoss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {'pamir_net': self.pamir_net,
                            'diffusion': self.diffusion_model}
        self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_net,
                                'optimizer_diffusion_model': self.optm_diffusion_model}

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        assert self.options.pretrained_gcmr_checkpoint is not None, 'You must provide a pretrained GCMR model!'
        self.load_pretrained_gcmr(self.options.pretrained_gcmr_checkpoint)
        if self.options.pretrained_pamir_net_checkpoint is not None:
            self.load_pretrained_pamir_net(self.options.pretrained_pamir_net_checkpoint)

        # read energy weights
        self.loss_weights = {
            'geo': self.options.weight_geo,
            'diff': 0.5,
        }

        logging.info('#trainable_params = %d' %
                     sum(p.numel() for p in self.pamir_net.parameters() if p.requires_grad))

        # meta results
        now = datetime.datetime.now()
        self.log_file_path = os.path.join(
            self.options.log_dir, 'log_%s.npz' % now.strftime('%Y_%m_%d_%H_%M_%S'))

        # diffusion
        # 制定每一步的beta
        self.betas = torch.linspace(-6, 6, self.num_steps)  # size:100
        self.betas = torch.sigmoid(self.betas) * (0.5e-2 - 1e-5) + 1e-5
        # beta是递增的，最小值为0.00001,最大值为0.005, sigmooid func
        # 像学习率一样的一个东西，而且是一个比较小的值，所以就有理由假设逆扩散过程也是一个高斯分布

        # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        self.alphas = 1 - self.betas  # size: 100
        self.alphas_prod = torch.cumprod(self.alphas, 0)  # size: 100
        # 就是让每一个都错一下位
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)  # p表示previous
        # alphas_prod[:-1] 表示取出 从0开始到倒数第二个值
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape == self.alphas_prod.shape == self.alphas_prod_p.shape == \
               self.alphas_bar_sqrt.shape == self.one_minus_alphas_bar_log.shape \
               == self.one_minus_alphas_bar_sqrt.shape

    def init_count(self):
        super(BaseTrainer, self).__init__()
        if self.checkpoint is None:
            self.epoch_count_diff = 0
            self.step_count_diff = 0
        else:
            self.epoch_count_diff = 0
            self.step_count_diff = 0
        #     self.epoch_count_diff = self.checkpoint['epoch_count_diff']
        #     self.step_count_diff = self.checkpoint['step_count_diff']

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch
            def worker_init_fn(worker_id):  # set numpy's random seed
                seed = torch.initial_seed()
                seed = seed % (2 ** 32)
                np.random.seed(seed + worker_id)

            self.start_epoch(epoch)
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
                                                     worker_init_fn=worker_init_fn)

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, out)

                    # Backup the current training stage
                    if self.step_count % (self.options.summary_steps*10) == 0:
                        self.saver.save_latest(
                            self.models_dict, self.optimizers_dict, epoch, step + 1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(
                            self.models_dict, self.optimizers_dict, epoch, step+1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(
                        self.models_dict, self.optimizers_dict, epoch, step,
                        self.options.batch_size, train_data_loader.sampler.dataset_perm,
                        self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(
                # self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(
                    self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size,
                    None, self.step_count)

        tqdm.write('Diffusion Training')
        # run diffusion training
        for epoch in tqdm(range(self.epoch_count_diff, self.options.num_epochs), total=self.options.num_epochs,
                          initial=self.epoch_count_diff):
            def worker_init_fn(worker_id):  # set numpy's random seed
                seed = torch.initial_seed()
                seed = seed % (2 ** 32)
                np.random.seed(seed + worker_id)

            self.start_epoch(epoch)
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
                                                     worker_init_fn=worker_init_fn)

            epoch_diff = epoch + self.epoch_count
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch ' + str(epoch_diff),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    with torch.autograd.set_detect_anomaly(True):
                        out = self.train_step_diff(batch)
                    self.step_count_diff += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count_diff % self.options.summary_steps == 0:
                        self.train_summaries(batch, out)

                    # Backup the current training stage
                    if self.step_count_diff % (self.options.summary_steps*10) == 0:
                        self.saver.save_latest(
                            self.models_dict, self.optimizers_dict, epoch_diff, step + 1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count_diff)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count_diff % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(
                            self.models_dict, self.optimizers_dict, epoch_diff, step+1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count_diff)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count_diff % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(
                        self.models_dict, self.optimizers_dict, epoch_diff, step,
                        self.options.batch_size, train_data_loader.sampler.dataset_perm,
                        self.step_count_diff)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each epoch_diff
            if (epoch_diff + 1) % 10 == 0:
                # self.saver.save_checkpoint(
                # self.models_dict, self.optimizers_dict, epoch_diff+1, 0, self.step_count_diff)
                self.saver.save_checkpoint(
                    self.models_dict, self.optimizers_dict, epoch_diff + 1, 0, self.options.batch_size,
                    None, self.step_count_diff)
        return

    def train_step_diff(self, input_batch):
        self.diffusion_model.train()
        self.pamir_net.eval()  # lock BN and dropout
        self.graph_cnn.eval()  # lock BN and dropout
        self.smpl_param_regressor.eval()  # lock BN and dropout

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        pts = input_batch['pts']    # [:, :-self.options.point_num]
        pts_proj = input_batch['pts_proj']  # [:, :-self.options.point_num]
        gt_ov = input_batch['pts_ov']   # [:, :-self.options.point_num]
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']
        pts2smpl_idx = input_batch['pts2smpl_idx']  # [:, :-self.options.point_num]
        pts2smpl_wgt = input_batch['pts2smpl_wgt']  # [:, :-self.options.point_num]

        batch_size, pts_num = pts.size()[:2]
        losses = dict()

        # prepare gt variables
        # convert to rotation matrices, add 180-degree rotation to root
        gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans

        # gcmr body prediction
        if not self.options.use_gt_smpl_volume:
            with torch.no_grad():
                pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
                pred_vert, pred_vert_tetsmpl, pred_keypoints_2d = self.forward_gcmr(img)

                # camera coordinate conversion
                scale_, trans_ = self.forward_coordinate_conversion(
                    pred_vert_tetsmpl, cam_f, cam_tz, cam_c, cam_r, cam_t, pred_cam, gt_trans)
                # pred_vert_cam = scale_ * pred_vert + trans_
                # pred_vert_tetsmpl_cam = scale_ * pred_vert_tetsmpl + trans_

                pred_vert_tetsmpl_gtshape_cam = \
                    scale_ * self.tet_smpl(pred_rotmat, pred_betas.detach()) + trans_

            # randomly replace one predicted SMPL with ground-truth one
            rand_id = np.random.randint(0, batch_size, size=[batch_size//3])
            rand_id = torch.from_numpy(rand_id).long()
            pred_vert_tetsmpl_gtshape_cam[rand_id] = gt_vert_cam[rand_id]

            if self.options.use_adaptive_geo_loss:
                pts = self.forward_warp_gt_field(
                    pred_vert_tetsmpl_gtshape_cam, gt_vert_cam, pts, pts2smpl_idx, pts2smpl_wgt)

        if self.options.use_gt_smpl_volume:
            vol = self.voxelization(gt_vert_cam)
        else:
            vol = self.voxelization(pred_vert_tetsmpl_gtshape_cam)

        output_sdf = self.pamir_net(img, vol, pts, pts_proj)

        # diffusion
        x_seq_list = []
        for pt_sdf in output_sdf:

            losses['diff'] = self.diffusion_model.diffusion_loss_fn(pt_sdf, self.alphas_bar_sqrt,
                                                          self.one_minus_alphas_bar_sqrt, self.num_steps)
            # self.optm_diffusion_model.zero_grad()
            # losses['diff'].backward(retain_graph=True)
            # 对梯度进行clip ， 保证稳定性.
            # 当神经网络深度逐渐增加，网络参数量增多的时候，反向传播过程中链式法则里的梯度连乘项数便会增多，
            # 更易引起梯度消失和梯度爆炸。对于梯度爆炸问题，解决方法之一便是进行梯度剪裁，即设置一个梯度大小的上限。

            # 梯度裁剪应该放在loss.backward()和optimizer.step()之间
            # https://zhuanlan.zhihu.com/p/557949443
            # torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.)
            # self.optm_diffusion_model.step()
            # ema会导致在这个s的数据集上训练速度变慢，所以不用ema
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         param.data = ema(name, param.data)

            x_seq = self.diffusion_model.p_sample_loop(pt_sdf.shape, self.num_steps, self.betas, self.one_minus_alphas_bar_sqrt)
            x_seq_list.append(x_seq[-1])

        output_sdf = x_seq_list

        losses['geo'] = self.geo_loss(output_sdf, gt_ov)

        # calculates total loss
        total_loss = 0
        for ln in losses.keys():
            w = self.loss_weights[ln] if ln in self.loss_weights else 1.0
            total_loss += w * losses[ln]
        losses.update({'total_loss': total_loss})

        # Do backprop
        self.optm_pamir_net.zero_grad()
        self.optm_diffusion_model.zero_grad()
        total_loss.backward()
        self.optm_pamir_net.step()
        self.optm_diffusion_model.step()

        # save
        self.write_logs(losses)

        # update learning rate
        if self.step_count % 10000 == 0:
            learning_rate = self.options.lr * (0.9 ** (self.step_count//10000))
            logging.info('Epoch %d, LR = %f' % (self.step_count, learning_rate))
            for param_group in self.optm_pamir_net.param_groups:
                param_group['lr'] = learning_rate
        return losses

    def train_step(self, input_batch):
        self.pamir_net.train()
        self.graph_cnn.eval()  # lock BN and dropout
        self.smpl_param_regressor.eval()  # lock BN and dropout

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        pts = input_batch['pts']    # [:, :-self.options.point_num]
        pts_proj = input_batch['pts_proj']  # [:, :-self.options.point_num]
        gt_ov = input_batch['pts_ov']   # [:, :-self.options.point_num]
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']
        pts2smpl_idx = input_batch['pts2smpl_idx']  # [:, :-self.options.point_num]
        pts2smpl_wgt = input_batch['pts2smpl_wgt']  # [:, :-self.options.point_num]

        batch_size, pts_num = pts.size()[:2]
        losses = dict()

        # prepare gt variables
        # convert to rotation matrices, add 180-degree rotation to root
        gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans

        # gcmr body prediction
        if not self.options.use_gt_smpl_volume:
            with torch.no_grad():
                pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
                pred_vert, pred_vert_tetsmpl, pred_keypoints_2d = self.forward_gcmr(img)

                # camera coordinate conversion
                scale_, trans_ = self.forward_coordinate_conversion(
                    pred_vert_tetsmpl, cam_f, cam_tz, cam_c, cam_r, cam_t, pred_cam, gt_trans)
                # pred_vert_cam = scale_ * pred_vert + trans_
                # pred_vert_tetsmpl_cam = scale_ * pred_vert_tetsmpl + trans_

                pred_vert_tetsmpl_gtshape_cam = \
                    scale_ * self.tet_smpl(pred_rotmat, pred_betas.detach()) + trans_

            # randomly replace one predicted SMPL with ground-truth one
            rand_id = np.random.randint(0, batch_size, size=[batch_size//3])
            rand_id = torch.from_numpy(rand_id).long()
            pred_vert_tetsmpl_gtshape_cam[rand_id] = gt_vert_cam[rand_id]

            if self.options.use_adaptive_geo_loss:
                pts = self.forward_warp_gt_field(
                    pred_vert_tetsmpl_gtshape_cam, gt_vert_cam, pts, pts2smpl_idx, pts2smpl_wgt)

        if self.options.use_gt_smpl_volume:
            vol = self.voxelization(gt_vert_cam)
        else:
            vol = self.voxelization(pred_vert_tetsmpl_gtshape_cam)

        output_sdf = self.pamir_net(img, vol, pts, pts_proj)

        losses['geo'] = self.geo_loss(output_sdf, gt_ov)

        # calculates total loss
        total_loss = 0
        for ln in losses.keys():
            w = self.loss_weights[ln] if ln in self.loss_weights else 1.0
            total_loss += w * losses[ln]
        losses.update({'total_loss': total_loss})

        # Do backprop
        self.optm_pamir_net.zero_grad()
        total_loss.backward()
        self.optm_pamir_net.step()

        # save
        self.write_logs(losses)

        # update learning rate
        if self.step_count % 10000 == 0:
            learning_rate = self.options.lr * (0.9 ** (self.step_count//10000))
            logging.info('Epoch %d, LR = %f' % (self.step_count, learning_rate))
            for param_group in self.optm_pamir_net.param_groups:
                param_group['lr'] = learning_rate
        return losses

    def calculate_gt_rotmat(self, gt_pose):
        gt_rotmat = rodrigues(gt_pose.view((-1, 3)))
        gt_rotmat = gt_rotmat.view((self.options.batch_size, -1, 3, 3))
        gt_rotmat[:, 0, 1:3, :] = gt_rotmat[:, 0, 1:3, :] * -1.0  # global rotation
        return gt_rotmat

    def forward_gcmr(self, img):
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert_sub = pred_vert_sub.transpose(1, 2)
        pred_vert = self.graph_mesh.upsample(pred_vert_sub)
        x = torch.cat(
            [pred_vert_sub, self.graph_mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)],
            dim=-1)
        pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vert_tetsmpl = self.tet_smpl(pred_rotmat, pred_betas)
        pred_keypoints = self.smpl.get_joints(pred_vert)
        pred_keypoints_2d = orthographic_projection(pred_keypoints, pred_cam)
        return pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
               pred_vert, pred_vert_tetsmpl, pred_keypoints_2d

    def forward_coordinate_conversion(self, pred_vert_tetsmpl, cam_f, cam_tz, cam_c,
                                      cam_r, cam_t, pred_cam, gt_trans):
        # calculates camera parameters
        with torch.no_grad():
            pred_smpl_joints = self.tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
            pred_root = pred_smpl_joints[:, 0:1, :]
            if gt_trans is not None:
                scale = pred_cam[:, 0:1] * cam_c * (cam_tz - gt_trans[:, 0, 2:3]) / cam_f
                trans_x = pred_cam[:, 1:2] * cam_c * (
                        cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
                trans_y = -pred_cam[:, 2:3] * cam_c * (
                        cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
                trans_z = gt_trans[:, 0, 2:3] + 2 * pred_root[:, 0, 2:3] * scale
            else:
                scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
                trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
                trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
                trans_z = torch.zeros_like(trans_x)
            scale_ = torch.cat([scale, -scale, -scale], dim=-1).detach().view((-1, 1, 3))
            trans_ = torch.cat([trans_x, trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))

        return scale_, trans_

    def forward_warp_gt_field(self, pred_vert_tetsmpl_gtshape_cam, gt_vert_cam,
                              pts, pts2smpl_idx, pts2smpl_wgt):
        with torch.no_grad():
            trans_gt2pred = pred_vert_tetsmpl_gtshape_cam - gt_vert_cam

            trans_z_pt_list = []
            for bi in range(self.options.batch_size):
                trans_pt_bi = (
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 0], 2] * pts2smpl_wgt[bi, :, 0] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 1], 2] * pts2smpl_wgt[bi, :, 1] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 2], 2] * pts2smpl_wgt[bi, :, 2] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 3], 2] * pts2smpl_wgt[bi, :, 3]
                )
                trans_z_pt_list.append(trans_pt_bi.unsqueeze(0))
            trans_z_pts = torch.cat(trans_z_pt_list, dim=0)
            # translate along z-axis to resolve depth inconsistency
            # pts[:, :, 2] += trans_z_pts
            pts[:, :, 2] += torch.tanh(trans_z_pts * 20) / 20
        return pts

    def forward_calculate_smpl_sub_vertice_in_cam(self, pred_vert_cam, cam_r, cam_t, cam_f, cam_c):
        pred_vert_sub_cam = self.graph_mesh.downsample(pred_vert_cam)
        pred_vert_sub_proj = pred_vert_sub_cam * cam_r.view((1, 1, -1)) + cam_t.view(
            (1, 1, -1))
        pred_vert_sub_proj = \
            pred_vert_sub_proj * (cam_f / cam_c) / pred_vert_sub_proj[:, :, 2:3]
        pred_vert_sub_proj = pred_vert_sub_proj[:, :, :2]
        return pred_vert_sub_cam, pred_vert_sub_proj

    def geo_loss(self, pred_ov, gt_ov):
        """Computes per-sample loss of the occupancy value"""
        if self.options.use_multistage_loss:
            loss = 0
            for o in pred_ov:
                loss = loss + self.criterion_geo(o, gt_ov)
        else:
            loss = self.criterion_geo(pred_ov[-1], gt_ov)
        return loss

    def train_summaries(self, input_batch, losses=None):
        assert losses is not None
        for ln in losses.keys():
            self.summary_writer.add_scalar(ln, losses[ln].item(), self.step_count)

    def write_logs(self, losses):
        data = dict()
        if os.path.exists(self.log_file_path):
            data = dict(np.load(self.log_file_path))
            for k in losses.keys():
                data[k] = np.append(data[k], losses[k].item())
        else:
            for k in losses.keys():
                data[k] = np.array([losses[k].item()])
        np.savez(self.log_file_path, **data)

    def load_pretrained_gcmr(self, model_path):
        if os.path.isdir(model_path):
            tmp = glob.glob(os.path.join(model_path, 'gcmr*.pt'))
            assert len(tmp) == 1
            logging.info('Loading GraphCMR from ' + tmp[0])
            data = torch.load(tmp[0])
        else:
            data = torch.load(model_path)
        self.graph_cnn.load_state_dict(data['graph_cnn'])
        self.smpl_param_regressor.load_state_dict(data['smpl_param_regressor'])

    def load_pretrained_pamir_net(self, model_path):
        if os.path.isdir(model_path):
            tmp1 = glob.glob(os.path.join(model_path, 'pamir_net*.pt'))
            assert len(tmp1) == 1
            logging.info('Loading pamir_net from ' + tmp1[0])
            data = torch.load(tmp1[0])
        else:
            data = torch.load(model_path)
        if 'pamir_net' in data:
            self.pamir_net.load_state_dict(data['pamir_net'])
        else:
            raise IOError('Failed to load pamir_net model from the specified checkpoint!!')
