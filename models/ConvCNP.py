import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from functools import partial
from einops import rearrange
import utils.misc as utils
from utils.metrics import WRMSE
from utils.builder import get_optimizer, get_lr_scheduler
import modules


class ConvCNP_model(nn.Module):

    def __init__(
        self,
        x_dim=69,
        y_dim=69,
        r_dim=512,
        XEncoder=nn.Identity,
        Conv=lambda y_dim: modules.make_abs_conv(nn.Conv2d)(
            y_dim,
            y_dim,
            groups=y_dim,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ),
        CNN=partial(
            modules.CNN,
            ConvBlock=modules.ResConvBlock,
            Conv=nn.Conv2d,
            n_blocks=12,
            Normalization=nn.BatchNorm2d,
            is_chan_last=True,
            kernel_size=9,
            n_conv_layers=2,
        ),
        PredictiveDistribution=modules.MultivariateNormalDiag,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=lambda y_scale: 0.01 + 0.99 * F.softplus(y_scale),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.conv = nn.ModuleList([Conv(y_dim), Conv(y_dim)])
        self.resizer = nn.ModuleList([nn.Linear(self.y_dim * 2, self.r_dim), nn.Linear(self.y_dim * 2, self.r_dim)])
        self.induced_to_induced = nn.ModuleList([CNN(self.r_dim), CNN(self.r_dim)])
        self.fusion = nn.Linear(self.r_dim * 2, self.r_dim)
        self.x_encoder = XEncoder()

        Decoder=modules.discard_ith_arg(partial(modules.MLP, n_hidden_layers=4, hidden_size=self.r_dim), i=0)
        # times 2 out because loc and scale (mean and var for gaussian)
        self.decoder = Decoder(self.x_dim, self.r_dim, self.y_dim * 2)

        self.PredictiveDistribution = PredictiveDistribution
        self.p_y_loc_transformer = p_y_loc_transformer
        self.p_y_scale_transformer = p_y_scale_transformer

        self.reset_parameters()

    def reset_parameters(self):
        modules.weights_init(self)

    def forward(self, input_list):

        X_cntxt, Y_cntxt, Xb_cntxt, Yb_cntxt, X_trgt = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
        X_cntxt = self.x_encoder(X_cntxt)  # b,h,w,c
        X_trgt = self.x_encoder(X_trgt)
        Xb_cntxt = self.x_encoder(Xb_cntxt)  # b,h,w,c

        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim] for n_channels list
        R = self.encode_globally(X_cntxt, Y_cntxt, Xb_cntxt, Yb_cntxt)

        z_samples, q_zCc, q_zCct = None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = self.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.decode(X_trgt, R_trgt)

        return p_yCc, z_samples, q_zCc, q_zCct

    def cntxt_to_induced(self, mask_cntxt, X, index):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        X = modules.channels_to_2nd_dim(X)
        # size = [batch_size, x_dim, *grid_shape]
        mask_cntxt = modules.channels_to_2nd_dim(mask_cntxt).float()

        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X * mask_cntxt
        signal = self.conv[index](X_cntxt)
        density = self.conv[index](mask_cntxt.expand_as(X))

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = modules.channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer[index](out)

        return out

    def encode_globally(self, mask_cntxt, X, mask_cntxtb, Xb):

        # size = [batch_size, *grid_shape, r_dim] for each single channel
        R_induced = self.cntxt_to_induced(mask_cntxt, X, index=0)
        R_induced = self.induced_to_induced[0](R_induced)

        Rb_induced = self.cntxt_to_induced(mask_cntxtb, Xb, index=1)
        Rb_induced = self.induced_to_induced[1](Rb_induced)

        R_induced = rearrange(R_induced, 'b h w c -> b c h w')
        R_induced = F.interpolate(R_induced, size=Rb_induced.shape[1:3], mode='bilinear')
        R_induced = rearrange(R_induced, 'b c h w -> b h w c')
        R_fusion = self.fusion(torch.cat([R_induced, Rb_induced], dim=-1))

        return R_fusion
    
    def trgt_dependent_representation(self, _, __, R_induced, ___):

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_induced.unsqueeze(0)
    
    def decode(self, X_trgt, R_trgt):

        # size = [n_z_samples, batch_size, *n_trgt, y_dim*2]
        p_y_suffstat = self.decoder(X_trgt, R_trgt)

        # size = [n_z_samples, batch_size, *n_trgt, y_dim]
        p_y_loc, p_y_scale = p_y_suffstat.split(self.y_dim, dim=-1)

        p_y_loc = self.p_y_loc_transformer(p_y_loc)
        p_y_scale = self.p_y_scale_transformer(p_y_scale)

        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.PredictiveDistribution(p_y_loc, p_y_scale)

        return p_yCc
    

class ConvCNP(object):
    
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        criterion = model_params.get('criterion', 'CNPFLoss')
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = ConvCNP_model(**params)
        self.best_loss = 9999
        self.criterion = self.get_criterion(criterion)
        self.criterion_mae = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()

        if utils.is_dist_avail_and_initialized():
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if self.device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_criterion(self, loss_type):
        if loss_type == 'CNPFLoss':
            return modules.CNPFLoss()
        elif loss_type == 'NLLLossLNPF':
            return modules.NLLLossLNPF()
        elif loss_type == 'ELBOLossLNPF':
            return modules.ELBOLossLNPF()
        elif loss_type == 'SUMOLossLNPF':
            return modules.SUMOLossLNPF()
        else:
            raise NotImplementedError('Invalid loss type.')
    
    def process_data(self, batch_data, args):

        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1)
        inp_data = F.interpolate(inp_data, size=(128,256), mode='bilinear').numpy()
        truth = batch_data[-1].to(self.device, non_blocking=True)  # 69
        truth = F.interpolate(truth, size=(args.resolution,args.resolution//2*4), mode='bilinear')
        truth_down = F.interpolate(truth, size=(128,256), mode='bilinear')

        for _ in range(args.lead_time // 6 + 2):
            predict_data = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
            inp_data = np.concatenate([inp_data[:,-truth.shape[1]:], predict_data], axis=1)        

        xb_context = rearrange(torch.rand(predict_data.shape, device=self.device) >= 0, 'b c h w -> b h w c')
        x_context = rearrange(torch.rand(truth.shape, device=self.device) >= args.ratio, 'b c h w -> b h w c')
        x_target = rearrange(torch.rand(truth_down.shape, device=self.device) >= 0, 'b c h w -> b h w c')
        yb_context = rearrange(torch.from_numpy(predict_data).to(self.device, non_blocking=True), 'b c h w -> b h w c')
        y_context = rearrange(truth, 'b c h w -> b h w c')
        y_target = rearrange(truth_down, 'b c h w -> b h w c')

        return [x_context, y_context, xb_context, yb_context, x_target], y_target
    
    def train(self, train_data_loader, valid_data_loader, logger, args):
        
        train_step = len(train_data_loader)
        valid_step = len(valid_data_loader)
        self.optimizer = get_optimizer(self.kernel, self.optimizer_params)
        self.scheduler = get_lr_scheduler(self.optimizer, self.scheduler_params, total_steps=train_step*args.max_epoch)

        for epoch in range(args.max_epoch):
            begin_time = time.time()
            self.kernel.train()
            
            for step, batch_data in enumerate(train_data_loader):

                input_list, y_target = self.process_data(batch_data[0], args)
                self.optimizer.zero_grad()
                y_pred = self.kernel(input_list)
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.train()
                loss = self.criterion(y_pred, y_target)
                loss.backward()
                clip_grad_norm_(self.kernel.parameters(), max_norm=1)
                self.optimizer.step()
                self.scheduler.step()
                
                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')

            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0

                for step, batch_data in enumerate(valid_data_loader):
                    input_list, y_target = self.process_data(batch_data[0], args)
                    y_pred = self.kernel(input_list)
                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.eval()
                    loss = self.criterion(y_pred, y_target).item()
                    total_loss += loss

                    if ((step + 1) % 100 == 0) | (step+1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}]')
        
            if (total_loss/valid_step) < self.best_loss:
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    torch.save(self.kernel.module.state_dict(), f'{args.rundir}/best_model.pth')
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.state_dict(), f'{args.rundir}/best_model.pth')
                logger.info(f'New best model appears in epoch {epoch+1}.')
                self.best_loss = total_loss/valid_step
            logger.info(f'Epoch {epoch+1} average loss:[{total_loss/valid_step}], time:[{time.time()-begin_time}]')

    def test(self, test_data_loader, logger, args):
        
        test_step = len(test_data_loader)
        data_mean, data_std = test_data_loader.dataset.get_meanstd()
        self.data_std = data_std.to(self.device)

        self.kernel.eval()
        with torch.no_grad():
            total_loss = 0
            total_mae = 0
            total_mse = 0
            total_rmse = 0

            for step, batch_data in enumerate(test_data_loader):

                input_list, y_target = self.process_data(batch_data[0], args)
                y_pred = self.kernel(input_list)
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.eval()
                loss = self.criterion(y_pred, y_target).item()
                
                y_pred = rearrange(y_pred[0].mean[0], 'b h w c -> b c h w')
                y_target = rearrange(y_target, 'b h w c -> b c h w')
                mae = self.criterion_mae(y_pred, y_target).item()
                mse = self.criterion_mse(y_pred, y_target).item()
                rmse = WRMSE(y_pred, y_target, self.data_std)

                total_loss += loss
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                if ((step + 1) % 100 == 0) | (step+1 == test_step):
                    logger.info(f'Valid step:[{step+1}/{test_step}], loss:[{loss}], MAE:[{mae}], MSE:[{mse}]')

        logger.info(f'Average loss:[{total_loss/test_step}], MAE:[{total_mae/test_step}], MSE:[{total_mse/test_step}]')
        logger.info(f'Average RMSE:[{total_rmse/test_step}]')
