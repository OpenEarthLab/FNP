import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from timm.models.layers import trunc_normal_
import utils.misc as utils
from utils.metrics import WRMSE
from functools import partial
from modules import AllPatchEmbed, PatchRecover, BasicLayer, SwinTransformerLayer
from utils.builder import get_optimizer, get_lr_scheduler



class Adas_model(nn.Module):
    def __init__(self, img_size=(69,128,256), dim=96, patch_size=(1,2,2), window_size=(2,4,8), depth=8, num_heads=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, ape=True, use_checkpoint=False):
        super().__init__()

        self.patchembed = AllPatchEmbed(img_size=img_size, embed_dim=dim, patch_size=patch_size, norm_layer=nn.LayerNorm)  # b,c,14,180,360
        self.patchunembed = PatchRecover(img_size=img_size, embed_dim=dim, patch_size=patch_size)
        self.patch_resolution = self.patchembed.patch_resolution

        self.layer1 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)  # s1
        self.layer2 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s2
        self.layer3 = BasicLayer(dim*4, kernel=3, padding=1, num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s3
        self.layer4 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s2
        self.layer5 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s1

        self.fusion = nn.Conv3d(dim*3, dim, kernel_size=(3,5,7), stride=1, padding=(1,2,3))

        # absolute position embedding
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, self.patch_resolution[0], self.patch_resolution[1], self.patch_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder = SwinTransformerLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, qkv_bias=True, 
                                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, use_checkpoint=use_checkpoint)

        # initial weights
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)
        x = self.layer4(x, x2)
        x = self.layer5(x, x1)

        return x

    def forward(self, background, observation, mask):

        x = self.patchembed(background, observation, mask)
        if self.ape:
            x = [ x[i] + self.absolute_pos_embed for i in range(3) ]

        x = self.encoder_forward(x)
        x = self.fusion(torch.cat(x, dim=1))
        x = self.decoder(x)

        x = self.patchunembed(x)
        return x
    

class Adas(object):
    
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        criterion = model_params.get('criterion', 'CNPFLoss')
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = Adas_model(**params)
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
        if loss_type == 'UnifyMAE':
            return partial(self.unify_losses, criterion=nn.L1Loss())
        elif loss_type == 'UnifyMSE':
            return partial(self.unify_losses, criterion=nn.MSELoss())
        else:
            raise NotImplementedError('Invalid loss type.')

    def unify_losses(self, pred, target, criterion):
        loss_sum = 0
        unify_loss = criterion(pred[:,0,:,:], target[:,0,:,:])
        for i in range(1, len(pred[0])):
            loss = criterion(pred[:,i,:,:], target[:,i,:,:])
            loss_sum += loss / (loss/unify_loss).detach()
        return (loss_sum + unify_loss) / len(pred[0])
    
    def process_data(self, batch_data, args):

        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1)
        inp_data = F.interpolate(inp_data, size=(128,256), mode='bilinear').numpy()
        truth = batch_data[-1].to(self.device, non_blocking=True)  # 69
        truth = F.interpolate(truth, size=(args.resolution,args.resolution//2*4), mode='bilinear')
        truth_down = F.interpolate(truth, size=(128,256), mode='bilinear')

        for _ in range(args.lead_time // 6):
            predict_data = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
            inp_data = np.concatenate([inp_data[:,-truth.shape[1]:], predict_data], axis=1)        

        background = torch.from_numpy(predict_data).to(self.device, non_blocking=True)
        mask = (torch.rand(truth.shape, device=self.device) >= args.ratio).float()
        observation = truth * mask
        mask = F.interpolate(mask, size=(128,256), mode='bilinear')
        observation = F.interpolate(observation, size=(128,256), mode='bilinear')
        observation = torch.where(mask==0, 0., observation/mask).to(self.device, non_blocking=True)
        mask = torch.where(mask==0, 0., 1.).to(self.device, non_blocking=True)

        return [background, observation, mask], truth_down
    
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
                y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
                loss = self.criterion(y_pred, y_target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')

            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0

                for step, batch_data in enumerate(valid_data_loader):
                    input_list, y_target = self.process_data(batch_data[0], args)
                    y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
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
                y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
                loss = self.criterion(y_pred, y_target).item()
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
