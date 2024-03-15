import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.functional.backend import _backend
from modules.functional.sampling import gather, furthest_point_sample
import modules.functional as mf
import pytorch3d
from ..utils.utils import normalize_to_box, sample_farthest_points
from modules.chamfer_distance import ChamferDistance

__all__ = ['YOGO']


def knn_search(input_pts, query_pts, k):
    knn_idx = knn.knn_batch(input_pts.permute(0, 2, 1).data.cpu(),
                            query_pts.permute(0, 2, 1).data.cpu(), k
                            )
    return knn_idx.astype(np.int64)


def conv1x1_1d(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

def conv1x1_2d(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)


def conv1x1(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)


class MatMul(nn.Module):
    """A wrapper class such that we can count the FLOPs of matmul
    """

    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)


class Transformer(nn.Module):
    def __init__(self, token_c, t_layer=1, head=2, kqv_groups=1,
                 norm_layer_1d=nn.Identity):
        super(Transformer, self).__init__()

        self.k_conv = nn.ModuleList()
        self.q_conv = nn.ModuleList()
        self.v_conv = nn.ModuleList()
        self.kqv_bn = nn.ModuleList()
        self.kq_matmul = nn.ModuleList()
        self.kqv_matmul = nn.ModuleList()
        self.ff_conv = nn.ModuleList()
        for _ in range(t_layer):
            self.k_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.q_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.v_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c, groups=kqv_groups),
                norm_layer_1d(token_c)
            ))
            self.kq_matmul.append(MatMul())
            self.kqv_matmul.append(MatMul())
            self.kqv_bn.append(norm_layer_1d(token_c))
            # zero-init
            # nn.init.constant_(self.kqv_bn[-1].weight, 0)
            self.ff_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c * 2),
                norm_layer_1d(token_c * 2),
                nn.ReLU(inplace=True),
                conv1x1_1d(token_c * 2, token_c),
                norm_layer_1d(token_c),
            ))
            # initialize the bn weight to zero to improves the training
            # stability.
            # nn.init.constant_(self.ff_conv[-1][1].weight, 1)

        self.token_c = token_c
        self.t_layer = t_layer
        self.head = head

    def forward(self, x):
        N = x.shape[0]
        for _idx in range(self.t_layer):
            k = self.k_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            q = self.q_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            v = self.v_conv[_idx](x).view(
                N, self.head, self.token_c // self.head, -1)
            # N, h, L, C/h * N, h, C/h, L -> N, h, L, L
            kq = self.kq_matmul[_idx](k.permute(0, 1, 3, 2), q)
            # N, h, L, L
            kq = F.softmax(kq / np.sqrt(self.token_c / 2 / self.head), dim=2)
            # N, h, C/h, L * N, h, L, L -> N, h, C/h, L
            kqv = self.kqv_matmul[_idx](v, kq).view(N, self.token_c, -1)
            kqv = self.kqv_bn[_idx](kqv)
            x = x + kqv
            x = x + self.ff_conv[_idx](x)

        return x


class Projector(nn.Module):
    def __init__(self, token_c, planes, head=2, min_group_planes=64,
                 norm_layer_1d=nn.Identity):
        super(Projector, self).__init__()

        if token_c != planes:
            self.proj_value_conv = nn.Sequential(
                conv1x1_1d(token_c, planes),
                norm_layer_1d(planes))
        else:
            self.proj_value_conv = nn.Identity()

        self.proj_key_conv = nn.Sequential(
            conv1x1_1d(token_c, planes),
            norm_layer_1d(planes)
        )
        self.proj_query_conv = nn.Sequential(
            conv1x1_1d(planes, planes),
            norm_layer_1d(planes)
        )
        self.proj_kq_matmul = MatMul()
        self.proj_matmul = MatMul()
        self.proj_bn = norm_layer_1d(planes)
        # zero-init
        # nn.init.constant_(self.proj_bn.weight, 1)

        self.ff_conv = nn.Sequential(
            conv1x1_1d(planes, 2 * planes),
            norm_layer_1d(2 * planes),
            nn.ReLU(inplace=True),
            conv1x1_1d(2 * planes, planes),
            norm_layer_1d(planes)
        )

        self.head = head

    def forward(self, x, x_t):
        N, _, L = x_t.shape
        h = self.head
        # -> N, h, C/h, L
        proj_v = self.proj_value_conv(x_t).view(N, h, -1, L)
        # -> N, h, C/h, L
        proj_k = self.proj_key_conv(x_t).view(N, h, -1, L)
        proj_q = self.proj_query_conv(x)
        N, C, _ = proj_q.shape
        # -> N, h, HW, c/H
        proj_q = proj_q.view(N, h, C // h, -1).permute(0, 1, 3, 2)
        # N, h, HW, C/h * N, h, C/h, L -> N, h, HW, L
        proj_coef = F.softmax(
            self.proj_kq_matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)

        # N, h, C/h, L * N, h, L, HW -> N, h, C/h, HW
        x_p = self.proj_matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        # -> N, C, H, W
        _, _, S = x.shape
        x_p = self.proj_bn(x_p.view(N, -1, S))

        x = x + self.ff_conv(x + x_p)

        return x


class RIM(nn.Module):
    def __init__(self, token_c, input_dims, output_dims,
                 head=2, min_group_planes=1, norm_layer_1d=nn.Identity,
                 **kwargs):
        super(RIM, self).__init__()

        self.transformer = Transformer(
            token_c, norm_layer_1d=norm_layer_1d, head=head,
            **kwargs)

        if input_dims == output_dims:
            self.feature_block = nn.Identity()
        else:
            self.feature_block = nn.Sequential(
                conv1x1_1d(input_dims, output_dims),
                norm_layer_1d(output_dims)
            )

        self.projectors = Projector(
            token_c, output_dims, head=head,
            min_group_planes=min_group_planes,
            norm_layer_1d=norm_layer_1d)

        self.dynamic_f = nn.Sequential(
            conv1x1_1d(input_dims, token_c),
            norm_layer_1d(token_c),
            nn.ReLU(inplace=True),
            conv1x1_1d(token_c, token_c),
            norm_layer_1d(token_c)
        )

    def forward(self, in_feature, in_tokens, knn_idx):
        # in_feature: B, N, C
        # in_coords: B, N, 3
        B, L, K = knn_idx.shape
        B, C, N = in_feature.shape

        gather_fts = gather(
            in_feature, knn_idx.view(B, -1)
        ).view(B, -1, L, K)

        tokens = self.dynamic_f(gather_fts.max(dim=3)[0])

        t_c = tokens.shape[1]

        if in_tokens is not None:
            tokens += in_tokens

        tokens = self.transformer(tokens)

        out_feature = self.projectors(
            self.feature_block(in_feature), tokens
        )

        return out_feature, tokens


class RIM_ResidualBlock(nn.Module):
    def __init__(self, inc, outc, token_c, norm_layer_1d):
        super(RIM_ResidualBlock, self).__init__()
        if inc != outc:
            self.res_connect = nn.Sequential(
                nn.Conv1d(inc, outc, 1),
                norm_layer_1d(outc),
            )
        else:
            self.res_connect = nn.Identity()
        self.vt1 = RIM(
            token_c, inc, inc, norm_layer_1d=norm_layer_1d)
        self.vt2 = RIM(
            token_c, inc, outc, norm_layer_1d=norm_layer_1d)

    def forward(self, inputs):
        in_feature, tokens, knn_idx = inputs
        out, tokens = self.vt1(in_feature, tokens, knn_idx)
        out, tokens = self.vt2(out, tokens, knn_idx)

        return out, tokens


class YOGO(nn.Module):
    def __init__(self, token_l, token_s, token_c, ball_r, ):
        super().__init__()
        cr = 1
        cs = [32, 64, 128, 128]
        cs = [int(cr * x) for x in cs]

        self.token_l = token_l
        self.token_s = token_s
        self.token_c = token_c

        self.group_ = 'ball_query'
        self.ball_r = ball_r

        norm_layer = nn.InstanceNorm1d

        self.stem = nn.Sequential(
            conv1x1_1d(3, cs[0]),
            norm_layer(cs[0]),
            nn.ReLU(inplace=True),
            conv1x1_1d(cs[0], cs[0]),
            norm_layer(cs[0])
        )

        self.stage1 = nn.Sequential(
            RIM_ResidualBlock(cs[0], cs[1], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage2 = nn.Sequential(
            RIM_ResidualBlock(cs[1], cs[2], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage3 = nn.Sequential(
            RIM_ResidualBlock(cs[2], cs[3], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        # self.stage4 = nn.Sequential(
        #     RIM_ResidualBlock(cs[3], cs[4], token_c=self.token_c, norm_layer_1d=norm_layer),
        # )

    def forward(self, x, keypoints, ):

        coords = x[:, :3, :]


        B, _, N = x.shape

        feature_stem = self.stem(x)
        center_pts = keypoints


        if self.group_ == 'ball_query':
            knn_idx = mf.ball_query(
                center_pts, coords, self.ball_r, self.token_s
            )
        else:
            knn_idx = knn_search(coords, center_pts, self.token_s)
            knn_idx = torch.from_numpy(knn_idx).cuda()

        feature1, tokens = self.stage1((feature_stem, None, knn_idx))

        feature2, tokens = self.stage2((feature1, tokens, knn_idx))

        feature3, tokens = self.stage3((feature2, tokens, knn_idx))

        return feature3, tokens, knn_idx


class Decoder(nn.Module):
    # [ B * (3+z) * N ] -> # [ B * 3 * N ]
    def __init__(self, token_l, token_s, token_c,):
        super(Decoder, self).__init__()
        input_dim = token_c + 3 + token_l * 3 + token_l
        self.head_list = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)  # no dropout
        self.token_l = token_l
        self.token_s = token_s
        self.token_c = token_c

        cs = [128, 64, 32, 3]
        norm_layer = nn.InstanceNorm2d

        self.layer1 = nn.Sequential(
            conv1x1_2d(input_dim, cs[0]),
            norm_layer(cs[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            conv1x1_2d(cs[0], cs[1]),
            norm_layer(cs[1]),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            conv1x1_2d(cs[1], cs[2]),
            norm_layer(cs[2]),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            conv1x1_2d(cs[2], cs[3]),
        )

    def forward(self, src, tokens, kp,):
        # tokens B * C * L
        # src_region_pts B * 3 * L * S
        B, _, L, S = src.shape
        assert L == self.token_l
        assert S == self.token_s
        kp = torch.flatten(kp, start_dim=1)
        tokens_rpt = tokens.unsqueeze(3).repeat(1,1,1,self.token_s)
        kp_rpt = kp.unsqueeze(2).unsqueeze(3).repeat(1,1,self.token_l,self.token_s)
        region_emb = torch.zeros([B, self.token_l, self.token_l, self.token_s]).to(src.device)
        for i in range(self.token_l):
            region_emb[:, i, i, :] = 1
        input_ft = torch.cat([src, tokens_rpt, kp_rpt, region_emb], dim=1)
        x = input_ft
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class LatentNetwork(nn.Module):
    def __init__(self, opt, ):
        super(LatentNetwork, self).__init__()
        self.opt = opt
        self.token_l = opt.n_keypoints
        self.token_s = opt.num_region_pts
        self.token_c = opt.token_c
        self.encoder = YOGO(self.token_l, self.token_s, self.token_c, opt.ball_r,)
        if self.opt.use_partial_retrieval:
            self.partial_encoder = YOGO(self.token_l, self.token_s, self.token_c, opt.ball_r, )
        else:
            self.partial_encoder = None
        self.decoder = Decoder(self.token_l, self.token_s, self.token_c,)
        self.init_optimizer()
        if opt.ico_recon:
            self.template_vertices, _ = self.create_cage()

    def init_optimizer(self):
        if self.opt.partial_pc and self.opt.use_partial_retrieval:
            params = [{"params": self.partial_encoder.parameters()}]
            self.encoder_optimizer = torch.optim.Adam(params, lr=self.opt.encoder_lr)
        else:
            params = [{"params": self.encoder.parameters()}]
            self.encoder_optimizer = torch.optim.Adam(params, lr=self.opt.encoder_lr)
        params = [{"params": self.decoder.parameters()}]
        self.decoder_optimizer = torch.optim.Adam(params, lr=self.opt.decoder_lr)

    def copy_param(self):
        self.partial_encoder.load_state_dict(self.encoder.state_dict())

    def create_cage(self, level=3):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(level, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.opt.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F


    def forward(self, src, tgt, src_kp, tgt_kp, src2tgt, is_partial=False):
        self.source = src
        self.target = tgt
        src_kp = src_kp.detach()
        tgt_kp = tgt_kp.detach() if tgt_kp is not None else None
        src2tgt = src2tgt.detach() if src2tgt is not None else None
        self.source_keypoint = src_kp
        self.target_keypoint = tgt_kp
        self.src2tgt = src2tgt
        if is_partial:
            src = torch.cat([src, src_kp], dim=-1)
        B, _, N = src.shape
        if is_partial and self.opt.use_partial_retrieval:
            feature, tokens, knn_idx = self.partial_encoder(src, src_kp, )
        else:
            feature, tokens, knn_idx = self.encoder(src, src_kp,)
        self.src_region_pts = gather(
            src, knn_idx.view(B, -1)
        ).view(B, -1, self.token_l, self.token_s)
        self.src2tgt_region_pts = gather(
            src2tgt, knn_idx.view(B, -1)
        ).view(B, -1, self.token_l, self.token_s) if src2tgt is not None else None
        if self.opt.ico_recon:
            sample_idx = torch.randperm(self.template_vertices.shape[2])[:self.token_s]
            ico_sample = self.template_vertices[:, :, sample_idx]
            ico_sample_rpt = ico_sample.unsqueeze(2).repeat(B, 1, self.token_l, 1)
            self.recon_src2tgt = self.decoder(ico_sample_rpt, tokens, self.target_keypoint) if tgt_kp is not None else None
            self.recon_src = self.decoder(ico_sample_rpt, tokens, self.source_keypoint)
        else:
            self.recon_src2tgt = self.decoder(self.src_region_pts, tokens, self.target_keypoint) if tgt_kp is not None else None
            self.recon_src = self.decoder(self.src_region_pts, tokens, self.source_keypoint)
        # tokens B * C * L
        # src_region_pts B * 3 * L * S
        # loss = self.compute_loss()
        outputs = {
            'recon_src2tgt': self.recon_src2tgt,
            'recon_src': self.recon_src,
            'knn_idx': knn_idx,
            'tokens': tokens
        }
        return outputs

    def encode(self, src, src_kp, is_partial=False):
        if is_partial and self.opt.use_partial_retrieval:
            feature, tokens, knn_idx = self.partial_encoder(src, src_kp, )
        else:
            feature, tokens, knn_idx = self.encoder(src, src_kp,)
        outputs = {
            'knn_idx': knn_idx,
            'tokens': tokens
        }
        return outputs

    def compute_loss(self, iteration, result=None):
        losses = {}
        if self.opt.partial_pc:
            unique_count = torch.unique(result['knn_idx'], dim=-1)
            mask = unique_count != unique_count[:, :, 0].unsqueeze(-1)
            weight = (mask.sum(-1).float() + 1) / self.token_s

            if self.opt.lambda_decoder_chamfer_src > 0:
                src_region_pts = self.src_region_pts.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)
                recon_src = self.recon_src.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)

                chamfer_loss = pytorch3d.loss.chamfer_distance(
                    src_region_pts, recon_src, batch_reduction=None)[0]
                weighted_chamfer_loss = weight.flatten() * chamfer_loss
                losses['chamfer_src'] = self.opt.lambda_decoder_chamfer_src * weighted_chamfer_loss.mean()

            if self.opt.lambda_full_token_l1 > 0:
                l1_loss = torch.abs(result['partial_tokens'] - result['full_tokens']).mean(dim=1)
                weighted_l1_loss = weight * l1_loss
                losses['full_l1'] = self.opt.lambda_full_token_l1 * weighted_l1_loss.mean()

        else:
            if self.opt.lambda_decoder_chamfer_src > 0:
                src_region_pts = self.src_region_pts.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)
                recon_src = self.recon_src.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)

                chamfer_loss = pytorch3d.loss.chamfer_distance(
                    src_region_pts, recon_src)[0]
                losses['chamfer_src'] = self.opt.lambda_decoder_chamfer_src * chamfer_loss

            if self.opt.lambda_decoder_chamfer_tgt > 0:
                src2tgt_region_pts = self.src2tgt_region_pts.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)
                recon_src2tgt = self.recon_src2tgt.permute(0, 2, 3, 1).reshape(-1, self.token_s, 3)

                chamfer_loss = pytorch3d.loss.chamfer_distance(
                    src2tgt_region_pts, recon_src2tgt)[0]
                losses['chamfer_tgt'] = self.opt.lambda_decoder_chamfer_tgt * chamfer_loss

        return losses

    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)

    def optimize(self, losses, iteration):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = self._sum_losses(losses, ['chamfer_src', 'chamfer_tgt', 'full_l1'])
        if loss > 0:
            loss.backward()
            self.encoder_optimizer.step()
            if not self.opt.partial_pc:
                self.decoder_optimizer.step()


def generate_valid_mask(knn_idx, valid_thr):
    unique_count = torch.unique(knn_idx, dim=-1)
    valid_mask = unique_count[:, :, valid_thr] > unique_count[:, :, 0]
    return valid_mask


def count_unique_num(knn_idx):
    unique_count = torch.unique(knn_idx, dim=-1)
    mask = unique_count != unique_count[:, :, 0].unsqueeze(-1)
    weight = (mask.sum(-1).float() + 1) / unique_count.shape[1]
    return weight
