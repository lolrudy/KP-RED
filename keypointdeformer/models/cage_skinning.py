import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops import rearrange

from ..utils.cages import deform_with_MVC, mean_value_coordinates_3D
from ..utils.networks import Linear, MLPDeformer2, PointNetfeat
from ..utils.utils import normalize_to_box, sample_farthest_points
from .latent_network import YOGO, conv1x1_1d, generate_valid_mask
from modules.chamfer_distance.chamfer_distance import ChamferDistance
from modules.functional.backend import _backend
import modules.functional as mf

class CageSkinning(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--lambda_init_points", type=float, help="", default=2.0)
        parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
        parser.add_argument("--lambda_influence_predict_l2", type=float, help="", default=1)
        parser.add_argument("--lambda_influence_predict_l1", type=float, help="", default=0.0)
        parser.add_argument("--iterations_init_points", type=float, help="", default=200)
        parser.add_argument("--iterations_init_end", type=float, help="", default=1000)
        parser.add_argument("--lambda_init_points_reduced", type=float, help="", default=0.1)
        parser.add_argument("--no_optimize_cage", action="store_true", help="")
        parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        parser.add_argument("--n_fps", type=int, help="")
        parser.add_argument("--influence_bias_lr", type=float, help="", default=0.1)
        parser.add_argument("--cat_target_token", type=bool, help="", default=False)
        parser.add_argument("--lambda_keypoint", type=float, help="", default=2.0)
        parser.add_argument("--lambda_valid_keypoint", type=float, help="", default=10.0)
        parser.add_argument("--update_influence", type=bool, help="", default=True)
        parser.add_argument("--use_full_tgt", type=bool, help="", default=False)
        parser.add_argument("--use_weighted_keypoint_loss", type=bool, help="", default=True)
        return parser

    
    def __init__(self, opt):
        super(CageSkinning, self).__init__()
        
        self.opt = opt
        self.dim = self.opt.dim
        self.token_l = opt.n_keypoints
        self.token_s = opt.num_region_pts
        self.token_c = opt.token_c

        
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices, template_faces)
        self.init_networks(opt.bottleneck_size, self.opt.dim, opt)
        self.init_optimizer()


    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.opt.ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.opt.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F

    def generate_keypoint_predictor(self, bottleneck_size, dim, opt):
        # keypoint predictor
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization))
        nd_decoder_kpt = MLPDeformer2(dim=dim, bottleneck_size=bottleneck_size, npoint=opt.n_keypoints,
                                residual=opt.d_residual, normalization=opt.normalization)
        return nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)
    

    def init_networks(self, bottleneck_size, dim, opt):
        # keypoint predictor
        self.keypoint_predictor = self.generate_keypoint_predictor(bottleneck_size, dim, opt)
        self.partial_keypoint_predictor = self.generate_keypoint_predictor(bottleneck_size, dim, opt)

        # influence predictor
        influence_size = self.template_vertices.shape[2]
        self.influence_size = influence_size
        self.shape_encoder = YOGO(self.token_l, self.token_s, self.token_c, self.opt.ball_r)
        if opt.cat_target_token:
            self.decoder_influence = nn.Sequential(
                Linear(self.token_c * 2 + self.token_l, influence_size, activation="lrelu",
                       normalization=opt.normalization),
                Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))

        else:
            self.decoder_influence = nn.Sequential(
                Linear(self.token_c + self.token_l, influence_size, activation="lrelu",
                       normalization=opt.normalization),
                Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))


    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.opt.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)


    def init_optimizer(self):
        params = [{"params": self.shape_encoder.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        self.optimizer.add_param_group({'params': self.decoder_influence.parameters(), 'lr': self.opt.lr})
        self.optimizer.add_param_group({'params': self.influence_param, 'lr': self.opt.influence_bias_lr})
        if self.opt.partial_pc:
            params = [{"params": self.partial_keypoint_predictor.parameters()}]
            self.keypoint_optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        else:
            params = [{"params": self.keypoint_predictor.parameters()}]
            self.keypoint_optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        self.chamfer_dist = ChamferDistance()


    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    
    def forward(self, source_shape, target_shape, full_target_shape=None):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        # import ipdb
        # ipdb.set_trace()

        if source_shape is None:
            if self.opt.partial_pc:
                target_keypoints = self.partial_keypoint_predictor(target_shape)
            else:
                target_keypoints = self.keypoint_predictor(target_shape)
            outputs = {
                "target_keypoints": target_keypoints,
            }
            return outputs

        B, _, _ = source_shape.shape
        self.target_shape = target_shape


        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape

        if self.opt.partial_pc:
            source_keypoints = self.keypoint_predictor(source_shape)
            target_keypoints = self.partial_keypoint_predictor(target_shape) if target_shape is not None else None
            keypoints = torch.cat([source_keypoints, target_keypoints], dim=0) if target_shape is not None else source_keypoints
            if full_target_shape is not None:
                full_target_keypoints = self.keypoint_predictor(full_target_shape)
                self.full_target_keypoints = torch.clamp(full_target_keypoints, -1.0, 1.0).detach()
                self.full_target_shape = full_target_shape
        else:
            keypoints = self.keypoint_predictor(shape)
        keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints
            target_keypoints = None

        self.shape = shape
        self.keypoints = keypoints
        self.target_keypoints = target_keypoints
        
        n_fps = self.opt.n_fps if self.opt.n_fps else 2 * self.opt.n_keypoints
        self.init_keypoints = sample_farthest_points(shape, n_fps)

        if target_shape is not None:
            source_init_keypoints, target_init_keypoints = torch.split(self.init_keypoints, B, dim=0)
        else:
            source_init_keypoints = self.init_keypoints
            target_init_keypoints = None

        cage = self.template_vertices
        if not self.opt.no_optimize_cage:
            # the initial cage is ico-sphere, pull cage vertices as close to the origin,
            # stop when distance to the shape is bellow the threshold
            cage = self.optimize_cage(cage, source_shape)

        outputs = {
            "cage": cage.transpose(1, 2),
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
            'source_init_keypoints': source_init_keypoints,
            'target_init_keypoints': target_init_keypoints
        }

        if target_shape is None:
            return outputs

        # influence = keypoint -> cage (C = I @ K)
        # the initial influence is zero-matrix
        self.influence = self.influence_param[None]
        feature, tokens, knn_idx = self.shape_encoder(source_shape, source_keypoints)
        region_emb = torch.zeros([B, self.token_l, self.token_l]).to(source_shape.device)
        for i in range(self.token_l):
            region_emb[:, i, i] = 1
        if self.opt.cat_target_token:
            _, target_tokens, _ = self.shape_encoder(target_shape, target_keypoints)
            input_feature = torch.cat([tokens, target_tokens, region_emb], dim=1).transpose(1, 2)
        else:
            input_feature = torch.cat([tokens, region_emb], dim=1).transpose(1, 2)

        self.influence_offset = self.decoder_influence(input_feature)
        self.influence = self.influence + self.influence_offset

        # filter out far-away keypoint's influence, only preserve top K (5) keypoints
        distance = torch.sum((source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.opt.n_influence_ratio)
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        if target_keypoints is not None:
            # transfer keypoint offset to cage offset
            base_cage = cage
            keypoints_offset = target_keypoints - source_keypoints
            cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
            new_cage = base_cage + cage_offset

            # transfer deformed cage to deformed mesh
            cage = cage.transpose(1, 2)
            new_cage = new_cage.transpose(1, 2)
            deformed_shapes, weights, _ = deform_with_MVC(
                cage, new_cage, self.template_faces.expand(B, -1, -1), source_shape.transpose(1, 2), verbose=True)

            # deform_with_MVC(cage, cage_deformed, cage_face, query, verbose=False)
            self.deformed_shapes = deformed_shapes
        else:
            cage = cage.transpose(1, 2)
            new_cage = None
            weights, weights_unnormed = mean_value_coordinates_3D(source_shape.transpose(1, 2), cage,
                                                                  self.template_faces.expand(B, -1, -1), verbose=True)
            self.deformed_shapes = None
        
        outputs.update({
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": self.deformed_shapes,
            "weight": weights,
            "influence": influence})
        
        return outputs

    def deform_with_cache(self, source_shape, target_shape, cache):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """

        B, _, _ = source_shape.shape

        target_keypoints = self.keypoint_predictor(target_shape)
        target_keypoints = torch.clamp(target_keypoints, -1.0, 1.0)

        cage, source_keypoints, influence, weights = cache['cage'], cache['source_keypoints'], cache['influence'], cache['weight']

        # import ipdb
        # ipdb.set_trace()

        outputs = {
            "cage": cage,
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
        }

        if (not self.opt.cache) and target_shape is None:
            return outputs

        cage = cage.transpose(1, 2)
        # transfer keypoint offset to cage offset
        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        # transfer deformed cage to deformed mesh
        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        weights = weights.detach()
        deformed_shapes = torch.sum(weights.unsqueeze(-1) * new_cage.unsqueeze(1), dim=2)

        outputs.update({
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": deformed_shapes,
            "weight": weights,
            "influence": influence})

        return outputs

    def compute_loss(self, iteration):
        losses = {}

        if self.opt.partial_pc and self.opt.lambda_keypoint > 0:
            if self.opt.lambda_valid_keypoint > 1:
                knn_idx = mf.ball_query(
                    self.full_target_keypoints, self.target_shape, self.opt.ball_r, self.opt.valid_threshold + 1
                )
                if self.opt.use_weighted_keypoint_loss:
                    full_target_keypoints = rearrange(self.full_target_keypoints, 'b d n -> b n d')
                    target_keypoints = rearrange(self.target_keypoints, 'b d n -> b n d')
                    l1_loss = torch.abs(full_target_keypoints - target_keypoints).mean(dim=2)
                    unique_count = torch.unique(knn_idx, dim=-1)
                    mask = unique_count != unique_count[:, :, 0].unsqueeze(-1)
                    weight = (mask.sum(-1).float() + 1) / self.token_s
                    weight = torch.clip(weight, min=0.1)
                    weighted_l1_loss = weight * l1_loss
                    losses['init_points'] = self.opt.lambda_keypoint * self.opt.lambda_valid_keypoint * weighted_l1_loss.mean()
                else:
                    valid_mask = generate_valid_mask(knn_idx, self.opt.valid_threshold)
                    valid_mask = valid_mask.detach()

                    l1_loss_func = torch.nn.L1Loss()
                    full_target_keypoints = rearrange(self.full_target_keypoints, 'b d n -> b n d')
                    target_keypoints = rearrange(self.target_keypoints, 'b d n -> b n d')
                    init_points_loss_valid = l1_loss_func(full_target_keypoints[valid_mask], target_keypoints[valid_mask])
                    init_points_loss_out = l1_loss_func(full_target_keypoints[~valid_mask], target_keypoints[~valid_mask])
                    losses['init_points'] = self.opt.lambda_keypoint * (self.opt.lambda_valid_keypoint * init_points_loss_valid +
                                                                        init_points_loss_out)
            else:
                l1_loss_func = torch.nn.L1Loss()
                full_target_keypoints = rearrange(self.full_target_keypoints, 'b d n -> b n d')
                target_keypoints = rearrange(self.target_keypoints, 'b d n -> b n d')
                init_points_loss_valid = l1_loss_func(full_target_keypoints, target_keypoints)
                losses['init_points'] = self.opt.lambda_keypoint * init_points_loss_valid

        if self.opt.lambda_init_points > 0 and not self.opt.partial_pc:
            init_points_loss = pytorch3d.loss.chamfer_distance(
                rearrange(self.keypoints, 'b d n -> b n d'),
                rearrange(self.init_keypoints, 'b d n -> b n d'))[0]
            losses['init_points'] = self.opt.lambda_init_points * init_points_loss


        if self.opt.lambda_chamfer > 0:
            if self.opt.partial_pc:
                if self.opt.use_full_tgt:
                    target_shape = self.full_target_shape
                else:
                    target_shape = self.target_shape
                dist1, dist2 = self.chamfer_dist(self.deformed_shapes, rearrange(target_shape, 'b d n -> b n d'))
                chamfer_loss = dist2.mean()
            else:
                chamfer_loss = pytorch3d.loss.chamfer_distance(
                    self.deformed_shapes, rearrange(self.target_shape, 'b d n -> b n d'))[0]
            losses['chamfer'] = self.opt.lambda_chamfer * chamfer_loss

        if self.opt.lambda_influence_predict_l2 > 0:
            losses['influence_predict_l2'] = self.opt.lambda_influence_predict_l2 * torch.mean(self.influence_offset ** 2)

        if self.opt.lambda_influence_predict_l1 > 0:
            losses['influence_predict_l1'] = self.opt.lambda_influence_predict_l1 * torch.mean(self.influence_offset.abs())

        return losses

    
    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)
        

    def optimize(self, losses, iteration):
        self.keypoint_optimizer.zero_grad()
        self.optimizer.zero_grad()

        if iteration == self.opt.iterations_init_end:
            print('reducing init-point loss')
            self.opt.lambda_init_points = self.opt.lambda_init_points_reduced

        if iteration < self.opt.iterations_init_points:
            keypoints_loss = self._sum_losses(losses, ['init_points'])
            keypoints_loss.backward(retain_graph=True)
            self.keypoint_optimizer.step()

        if iteration >= self.opt.iterations_init_points:
            loss = self._sum_losses(losses, ['chamfer', 'influence_predict_l2', 'init_points'])
            loss.backward()
            if self.opt.update_influence:
                self.optimizer.step()
            self.keypoint_optimizer.step()

