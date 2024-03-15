import json
import os
import time
from datetime import datetime

import numpy as np
import pytorch3d.io
import torch
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append('./')
from keypointdeformer.datasets import get_dataset
from keypointdeformer.models import get_model
from keypointdeformer.options.base_options import BaseOptions
from keypointdeformer.utils import io
from keypointdeformer.utils.cages import deform_with_MVC
from keypointdeformer.utils.nn import load_network, save_network, weights_init
from keypointdeformer.utils.utils import Timer
from tensorboardX import SummaryWriter
from keypointdeformer.utils import logger
from keypointdeformer.models.latent_network import LatentNetwork, generate_valid_mask
from tqdm import tqdm
import pytorch3d
from modules.chamfer_distance.chamfer_distance import ChamferDistance
import pdb

CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINT_EXT = '.pth'


def write_losses(writer, losses, step):
    for name, value in losses.items():
        writer.add_scalar('loss/' + name, value, global_step=step)


def save_normalization(file_path, center, scale):
    with open(file_path, 'w') as f:
        json.dump({'center': [str(x) for x in center.cpu().numpy()], 'scale': str(scale.cpu().numpy()[0])}, f)


def save_data_keypoints(data, save_dir, name):
    if name in data:
        io.save_keypoints(os.path.join(save_dir, name + '.txt'), data[name])


def save_data_txt(f, data, fmt):
    np.savetxt(f, data.cpu().detach().numpy(), fmt=fmt)


def save_pts(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    io.save_pts(f, points.cpu().detach().numpy(), normals=normals)


def save_ply(f, verts, faces):
    pytorch3d.io.save_ply(f, verts.cpu(), faces=faces.cpu())


def save_output(save_dir_root, data, outputs, save_mesh=True, save_auxilary=True):
    name = data['target_file']
    save_dir = os.path.join(save_dir_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # save meshes
    if save_mesh and 'source_mesh' in data:
        io.save_mesh(os.path.join(save_dir, 'source_mesh.obj'), data["source_mesh"], data["source_face"])

        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'source_vertices.txt'), data["source_mesh"], '%0.6f')
            save_data_txt(os.path.join(save_dir, 'source_faces.txt'), data["source_face"], '%d')

        if "target_mesh" in data:
            io.save_mesh(os.path.join(save_dir, 'target_mesh.obj'), data["target_mesh"], data["target_face"])

        if outputs is not None:
            deformed, weights, _ = deform_with_MVC(
                outputs["cage"][None], outputs["new_cage"][None], outputs["cage_face"][None],
                data["source_mesh"][None], verbose=True)
            io.save_mesh(os.path.join(save_dir, 'deformed_mesh.obj'), deformed[0], data["source_face"])
            if save_auxilary:
                save_data_txt(os.path.join(save_dir, 'weights.txt'), weights[0], '%0.6f')

    # save pointclouds
    save_pts(os.path.join(save_dir, 'source_pointcloud.pts'), data['source_shape'], normals=data['source_normals'])
    if outputs is not None:
        save_pts(os.path.join(save_dir, 'deformed_pointcloud.pts'), outputs['deformed'])
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'influence.txt'), outputs['influence'], '%0.6f')

    if 'target_normals' in data:
        save_pts(os.path.join(save_dir, 'target_pointcloud.pts'), data['target_shape'], normals=data['target_normals'])
    else:
        save_pts(os.path.join(save_dir, 'target_pointcloud.pts'), data['target_shape'])

    # save cages
    if outputs is not None:
        save_ply(os.path.join(save_dir, 'cage.ply'), outputs["cage"], outputs["cage_face"])
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'cage.txt'), outputs["cage"], '%0.6f')
        save_ply(os.path.join(save_dir, 'deformed_cage.ply'), outputs["new_cage"], outputs["cage_face"])

        if outputs is not None:
            io.save_keypoints(os.path.join(save_dir, 'source_keypoints.txt'),
                              outputs["source_keypoints"].transpose(0, 1))
            io.save_keypoints(os.path.join(save_dir, 'target_keypoints.txt'),
                              outputs["target_keypoints"].transpose(0, 1))

        save_data_keypoints(data, save_dir, 'source_keypoints_gt')
        save_data_keypoints(data, save_dir, 'target_keypoints_gt')

        if 'source_init_keypoints' in outputs:
            io.save_keypoints(os.path.join(save_dir, 'source_init_keypoints.txt'),
                              outputs['source_init_keypoints'].transpose(0, 1))
            io.save_keypoints(os.path.join(save_dir, 'target_init_keypoints.txt'),
                              outputs['target_init_keypoints'].transpose(0, 1))

        if 'source_keypoints_gt_center' in data:
            save_normalization(os.path.join(save_dir, 'source_keypoints_gt_normalization.txt'),
                               data['source_keypoints_gt_center'], data['source_keypoints_gt_scale'])

        if 'source_seg_points' in data:
            io.save_labelled_pointcloud(os.path.join(save_dir, 'source_seg_points.xyzrgb'),
                                        data['source_seg_points'].detach().cpu().numpy(),
                                        data['source_seg_labels'].detach().cpu().numpy())
            save_data_txt(os.path.join(save_dir, 'source_seg_labels.txt'), data['source_seg_labels'], '%d')

    if 'cd_loss' in outputs:
        save_data_txt(os.path.join(save_dir, 'cd_loss.txt'), outputs['cd_loss'].unsqueeze(0), '%0.6f')


def split_batch(data, b, singleton_keys=[]):
    return {k: v[b] if k not in singleton_keys else v[0] for k, v in data.items()}


def save_outputs(outputs_save_dir, data, outputs, save_mesh=True):
    for b in range(data['source_shape'].shape[0]):
        save_output(
            outputs_save_dir, split_batch(data, b, singleton_keys=['cage_face']),
            split_batch(outputs, b, singleton_keys=['cage_face']), save_mesh=save_mesh)


def save_tensor(outputs_save_dir, tensor, name):
    torch.save(tensor.cpu(), os.path.join(outputs_save_dir, f'{name}.pt'))


def load_tensor(outputs_save_dir, name):
    return torch.load(os.path.join(outputs_save_dir, f'{name}.pt'))


def get_data(dataset, data, use_partial=False):
    data = dataset.uncollate(data)

    if "target_shape" in data and "source_shape" in data:
        source_shape, target_shape = data["source_shape"], data["target_shape"]
        source_shape_t = source_shape.transpose(1, 2)
        target_shape_t = target_shape.transpose(1, 2)
    elif "source_shape" in data:
        source_shape = data["source_shape"]
        source_shape_t = source_shape.transpose(1, 2)
        target_shape_t = None
    elif "target_shape" in data:
        target_shape = data["target_shape"]
        source_shape_t = None
        target_shape_t = target_shape.transpose(1, 2)
    else:
        raise Exception('NO SHAPE IN DATA!')

    if use_partial:
        if "target_partial_shape" in data:
            target_partial_shape = data["target_partial_shape"]
            full_target_shape_t = target_shape_t.clone()
            target_shape_t = target_partial_shape.transpose(1, 2)
        else:
            full_target_shape_t = None
        return source_shape_t, target_shape_t, full_target_shape_t
    else:
        return source_shape_t, target_shape_t


def merge_data(datas, data):
    if datas is None:
        datas = {}
        for k, v in data.items():
            if v is not None:
                datas[k] = v
    else:
        for k, v in data.items():
            if isinstance(datas[k], list):
                datas[k].extend(v)
            else:
                datas[k] = torch.cat([datas[k], v], dim=0)
    return datas


def test(opt, save_subdir="test"):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt)

    dataset.load_source = True
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
        collate_fn=dataset.collate,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    deformer = get_model(opt.model)(opt).cuda()
    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
    load_network(deformer, ckpt)

    deformer.eval()

    latent_network = LatentNetwork(opt).cuda()
    ckpt = opt.latent_ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
    load_network(latent_network, ckpt)
    latent_network.eval()

    test_output_dir = os.path.join(log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    with torch.no_grad():
        token_len = opt.token_c * opt.n_keypoints
        source_tokens = torch.zeros([0, token_len]).cuda()
        source_shapes = torch.zeros([0, 3, opt.num_point]).cuda()
        source_datas = None
        source_outputs = None
        logger.info('generating latent codes for source shapes')
        for data in tqdm(dataloader):
            # data
            data = dataset.uncollate(data)
            source_shape_t, _ = get_data(dataset, data)
            outputs = deformer(source_shape_t, None)
            src_kp = outputs['source_keypoints']
            latent_outputs = latent_network.encode(source_shape_t, src_kp)
            tokens = latent_outputs['tokens'].flatten(start_dim=1)
            source_tokens = torch.cat([source_tokens, tokens], dim=0)
            source_shapes = torch.cat([source_shapes, source_shape_t], dim=0)
            source_datas = merge_data(source_datas, data)
            if opt.cache:
                outputs.pop('cage_face')
                source_outputs = merge_data(source_outputs, outputs)
            # pdb.set_trace()

        save_tensor(test_output_dir, source_tokens, 'source_tokens')
        save_tensor(test_output_dir, source_shapes, 'source_shapes')
        torch.save(source_datas['source_file'], os.path.join(test_output_dir, 'source_files.pkl'))

        dataset.load_source = False

        cd_loss_all = torch.zeros([0]).cuda()

        chamfer_dist = ChamferDistance()

        if opt.partial_pc and not opt.retrieval_full_shape and not opt.retrieval_full_token:
            source_tokens = source_tokens.reshape(-1, opt.token_c, opt.n_keypoints)

        logger.info('retrieval and deformation')
        for data in tqdm(dataloader):
            data = dataset.uncollate(data)

            _, target_shape_t, full_target_shape_t = get_data(dataset, data, True)
            if opt.partial_pc and opt.retrieval_full_shape:
                partial_target_shape_t = target_shape_t
                target_shape_t = full_target_shape_t
            outputs = deformer(None, target_shape_t)
            tgt_kp = outputs['target_keypoints']
            latent_outputs = latent_network.encode(target_shape_t, tgt_kp, is_partial=opt.partial_pc)
            #opt.partial_pc and not opt.retrieval_full_shape)
            target_tokens = latent_outputs['tokens']

            sel_indices = torch.zeros([0, opt.top_k], dtype=torch.long).cuda()
            i = -1
            for target_token_ in target_tokens:
                i += 1
                if opt.partial_pc and not opt.retrieval_full_shape and not opt.retrieval_full_token:
                    unique_count = torch.unique(latent_outputs['knn_idx'], dim=-1)
                    unique_count_ = unique_count[i]
                    mask = unique_count_ != unique_count_[:,0].unsqueeze(-1)
                    weight = mask.sum(-1).float() #+ 1
                    # weight[weight<32] = 0
                    # weight = torch.clip(weight, max=96, min=32)
                    # import ipdb
                    # ipdb.set_trace()
                    if weight.sum() < 32:
                        weight[:] = 1
                    weight = torch.nn.functional.normalize(weight, p=1, dim=0)
                    dist = torch.norm(source_tokens - target_token_, dim=1, p=1)
                    dist = torch.inner(dist, weight)
                    # mask = valid_mask[i]
                    # if mask.sum() >= 1:
                    #     target_token_ = target_token_[:, mask].flatten()
                    #     source_tokens_ = source_tokens[:, :, mask].flatten(start_dim=1)
                    # else:
                    #     target_token_ = target_token_.flatten()
                    #     source_tokens_ = source_tokens.flatten(start_dim=1)
                    # dist = torch.norm(source_tokens_ - target_token_, dim=1, p=None)
                else:
                    target_token_ = target_token_.flatten()
                    dist = torch.norm(source_tokens - target_token_, dim=1, p=1)
                knn = dist.topk(opt.top_k, largest=False)
                # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
                index = knn.indices
                sel_indices = torch.cat([sel_indices, index.unsqueeze(0)], dim=0)

            min_error = None
            final_indices = None
            best_outputs = None
            if opt.partial_pc and opt.retrieval_full_shape:
                target_shape_t = partial_target_shape_t
            for k in range(opt.top_k):
                source_indices = sel_indices[:, k]
                source_shape_t = source_shapes[source_indices]

                if opt.cache:
                    cache = {}
                    for key in source_outputs.keys():
                        cache[key] = source_outputs[key][source_indices]
                    outputs = deformer.deform_with_cache(source_shape_t, target_shape=target_shape_t, cache=cache)
                else:
                    outputs = deformer(source_shape_t, target_shape=target_shape_t)

                if opt.partial_pc:
                    dist1, dist2 = chamfer_dist(outputs['deformed'], target_shape_t.transpose(1, 2))
                    cur_error = dist2.mean(dim=-1)
                else:
                    cur_error = pytorch3d.loss.chamfer_distance(outputs['deformed'], target_shape_t.transpose(1, 2),
                                                                batch_reduction=None)[0]
                scale = data['target_scale'].float().squeeze(1).squeeze(1)
                cur_error = cur_error * scale * scale
                if min_error is None:
                    min_error = cur_error
                    final_indices = source_indices
                    best_outputs = outputs
                else:
                    mask = cur_error < min_error
                    final_indices[mask] = source_indices[mask]
                    min_error[mask] = cur_error[mask]
                    for key in best_outputs.keys():
                        if key == 'cage_face':
                            continue
                        best_outputs[key][mask] = outputs[key][mask]

            cd_loss_all = torch.cat([cd_loss_all, min_error])
            for key, value in source_datas.items():
                if isinstance(value, list):
                    data[key] = [value[i] for i in final_indices.cpu().tolist()]
                else:
                    data[key] = value[final_indices]
            best_outputs['cd_loss'] = min_error
            save_outputs(os.path.join(log_dir, save_subdir), data, best_outputs)
        logger.info(f'chamfer distance loss mean={cd_loss_all.mean()}, std={cd_loss_all.std()}')


def refine_deform(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate,
        num_workers=opt.n_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    source_dataset = get_dataset(opt.dataset)(opt)
    source_dataset.load_source = True
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
        collate_fn=dataset.collate,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    deformer = get_model(opt.model)(opt).cuda()
    deformer.apply(weights_init)
    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(deformer, ckpt)
    deformer.train()

    latent_network = LatentNetwork(opt).cuda()
    if opt.latent_ckpt:
        ckpt = opt.latent_ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(latent_network, ckpt)
    latent_network.eval()

    t = 0
    e = 0

    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(deformer) + "\n")
    summary_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir=os.path.join(checkpoints_dir, 'logs', summary_dir), flush_secs=5)

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    while t <= opt.n_iterations:
        if e % opt.encode_interval == 0:
            with torch.no_grad():
                logger.info('encoding source shapes')
                token_len = opt.token_c * opt.n_keypoints
                source_tokens = torch.zeros([0, token_len]).cuda()
                source_shapes = torch.zeros([0, 3, opt.num_point]).cuda()
                source_datas = None
                logger.info('generating latent codes for source shapes')
                for src_data_ in tqdm(source_dataloader):
                    # data
                    src_data_ = dataset.uncollate(src_data_)
                    source_shape_t, _ = get_data(dataset, src_data_)
                    outputs = deformer(source_shape_t, None)
                    src_kp = outputs['source_keypoints']
                    latent_outputs = latent_network.encode(source_shape_t, src_kp)
                    tokens = latent_outputs['tokens'].flatten(start_dim=1)
                    source_tokens = torch.cat([source_tokens, tokens], dim=0)
                    source_shapes = torch.cat([source_shapes, source_shape_t], dim=0)
                    source_datas = merge_data(source_datas, src_data_)
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break
            source_shape_t, target_shape_t, full_target_shape_t = get_data(dataset, data, True)
            with torch.no_grad():
                outputs = deformer(None, target_shape_t)
                tgt_kp = outputs['target_keypoints']
                latent_outputs = latent_network.encode(target_shape_t, tgt_kp)
                target_tokens = latent_outputs['tokens'].flatten(start_dim=1)

                sel_indices = torch.zeros([0, opt.top_k], dtype=torch.long).cuda()
                for target_token_ in target_tokens:
                    dist = torch.norm(source_tokens - target_token_, dim=1, p=None)
                    knn = dist.topk(opt.top_k, largest=False)
                    # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
                    index = knn.indices
                    sel_indices = torch.cat([sel_indices, index.unsqueeze(0)], dim=0)
            B = target_shape_t.shape[0]
            rand_idx = torch.randint(opt.top_k, (B,))
            source_indices = sel_indices[range(B), rand_idx]
            source_shape_t = source_shapes[source_indices]

            outputs = deformer(source_shape_t, target_shape=target_shape_t, full_target_shape=full_target_shape_t)
            current_loss = deformer.compute_loss(t)
            deformer.optimize(current_loss, t)

            if t % opt.save_interval == 0:
                outputs_save_dir = os.path.join(checkpoints_dir, 'outputs', '%07d' % t)
                save_outputs(outputs_save_dir, data, outputs, save_mesh=False)
                # save_network(latent_network, checkpoints_dir, network_label="latent_net", epoch_label=t)
                save_network(deformer, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()
            if (t % opt.log_interval == 0):
                samples_sec = opt.batch_size / iter_time
                losses_str = ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str)

                logger.info(log_str)
                log_file.write(log_str + "\n")

                write_losses(writer, current_loss, t)
            t += 1
        e += 1
    log_file.close()
    save_network(deformer, checkpoints_dir, network_label="net", epoch_label="final")
    save_network(latent_network, checkpoints_dir, network_label="latent_net", epoch_label="final")


def train_retrieval(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate,
        num_workers=opt.n_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    deformer = get_model(opt.model)(opt).cuda()
    deformer.apply(weights_init)
    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(deformer, ckpt)

    if opt.train_deformer:
        deformer.train()
        raise NotImplementedError
    else:
        deformer.eval()

    latent_network = LatentNetwork(opt).cuda()
    if opt.latent_ckpt:
        ckpt = opt.latent_ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(latent_network, ckpt)
        latent_network.copy_param()
    latent_network.train()

    t = 0

    # train
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(deformer) + "\n")
    summary_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir=os.path.join(checkpoints_dir, 'logs', summary_dir), flush_secs=5)

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    while t <= opt.n_iterations:
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            source_shape_t, target_shape_t, full_target_shape_t = get_data(dataset, data, True)
            if opt.partial_pc:
                with torch.no_grad():
                    full_outputs = deformer(full_target_shape_t, target_shape=source_shape_t)
                    full_decoder_outputs = latent_network(full_target_shape_t, source_shape_t, full_outputs['source_keypoints'],
                                                         full_outputs['target_keypoints'],
                                                         full_outputs['deformed'].transpose(1, 2))
                with torch.no_grad():
                    outputs = deformer(target_shape_t, target_shape=None)
                decoder_outputs = latent_network(target_shape_t, None, outputs['source_keypoints'],
                                                 None, None, is_partial=True)
                result = {}
                result['full_tokens'] = full_decoder_outputs['tokens']
                result['partial_tokens'] = decoder_outputs['tokens']
                result['knn_idx'] = decoder_outputs['knn_idx']

            else:
                with torch.no_grad():
                    outputs = deformer(source_shape_t, target_shape=target_shape_t)
                decoder_outputs = latent_network(source_shape_t, target_shape_t, outputs['source_keypoints'],
                                                 outputs['target_keypoints'],
                                                 outputs['deformed'].transpose(1, 2))
                result = None

            current_loss = latent_network.compute_loss(t, result)
            latent_network.optimize(current_loss, t)

            if opt.train_deformer:
                current_loss_deformer = deformer.compute_loss(t)
                deformer.optimize(current_loss_deformer, t)
                current_loss.update(current_loss_deformer)

            if t % opt.save_interval == 0:
                outputs_save_dir = os.path.join(checkpoints_dir, 'outputs', '%07d' % t)
                # save_outputs(outputs_save_dir, data, outputs, save_mesh=False)
                save_network(latent_network, checkpoints_dir, network_label="latent_net", epoch_label=t)
                if opt.train_deformer:
                    save_network(deformer, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()
            if (t % opt.log_interval == 0):
                samples_sec = opt.batch_size / iter_time
                losses_str = ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str)

                logger.info(log_str)
                log_file.write(log_str + "\n")

                write_losses(writer, current_loss, t)
            t += 1

    log_file.close()
    save_network(deformer, checkpoints_dir, network_label="net", epoch_label="final")
    save_network(latent_network, checkpoints_dir, network_label="latent_net", epoch_label="final")


def train(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate,
        num_workers=opt.n_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    net = get_model(opt.model)(opt).cuda()
    net.apply(weights_init)
    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    # train
    net.train()
    t = 0

    # train
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")
    summary_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir=os.path.join(checkpoints_dir, 'logs', summary_dir), flush_secs=5)

    if opt.iteration:
        t = opt.iteration

    iter_time_start = time.time()

    while t <= opt.n_iterations:
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            source_shape_t, target_shape_t, full_target_shape_t = get_data(dataset, data, True)
            outputs = net(source_shape_t, target_shape=target_shape_t, full_target_shape=full_target_shape_t)
            current_loss = net.compute_loss(t)
            net.optimize(current_loss, t)

            if t % opt.save_interval == 0:
                outputs_save_dir = os.path.join(checkpoints_dir, 'outputs', '%07d' % t)
                save_outputs(outputs_save_dir, data, outputs, save_mesh=False)
                save_network(net, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()
            if (t % opt.log_interval == 0):
                samples_sec = opt.batch_size / iter_time
                losses_str = ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str)

                logger.info(log_str)
                log_file.write(log_str + "\n")

                write_losses(writer, current_loss, t)
            t += 1

    log_file.close()
    save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        test(opt, save_subdir=opt.subdir)
    elif opt.phase == "refine_deform":
        refine_deform(opt)
    elif opt.phase == "train_retrieval":
        train_retrieval(opt)
    elif opt.phase == 'train':
        train(opt)
    else:
        raise ValueError()
