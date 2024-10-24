from torch.utils.data.distributed import DistributedSampler
from utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized
import onnxruntime as ort


class ConfigBuilder(object):
    """
    Configuration Builder.

    """
    def __init__(self, **params):
        """
        Set the default configuration for the configuration builder.

        Parameters
        ----------
        
        params: the configuration parameters.
        """
        super(ConfigBuilder, self).__init__()
        self.model_params = params.get('model', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})

        self.logger = params.get('logger', None)
    
    def get_model(self, model_params = None):
        """
        Get the model from configuration.

        Parameters
        ----------
        
        model_params: dict, optional, default: None. If model_params is provided, then use the parameters specified in the model_params to build the model. Otherwise, the model parameters in the self.params will be used to build the model.
        """
        from models.FNP import FNP
        from models.ConvCNP import ConvCNP
        from models.Adas import Adas

        if model_params is None:
            model_params = self.model_params
        type = model_params.get('type', 'FNP')

        if type == 'FNP':
            model = FNP(**model_params)
        elif type == 'ConvCNP':
            model = ConvCNP(**model_params)
        elif type == 'Adas':
            model = Adas(**model_params)
        else:
            raise NotImplementedError('Invalid model type.')
        
        return model
    
    def get_forecast(self, local_rank):

        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = 1

        # Set the behavier of cuda provider
        cuda_provider_options = {'device_id': local_rank, 'arena_extend_strategy':'kSameAsRequested',}

        # Initialize onnxruntime session for Pangu-Weather Models
        ort_session = ort.InferenceSession('./models/FengWu.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
        
        return ort_session

    def get_dataset(self, dataset_params = None, split = 'train'):
        """
        Get the dataset from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset.

        Returns
        -------
        
        A torch.utils.data.Dataset item.
        """
        from datasets.era5_npy_f32 import era5_npy_f32
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, None)
        if dataset_params is None:
            return None
        dataset = era5_npy_f32(split = split, **dataset_params)
        return dataset
    
    def get_sampler(self, dataset, split = 'train', drop_last=False):
        if split == 'train':
            shuffle = True
        else:
            shuffle = False
            
        if is_dist_avail_and_initialized():
            rank = get_rank()
            num_gpus = get_world_size()
        else:
            rank = 0
            num_gpus = 1
        sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, drop_last=drop_last)

        return sampler
   

    def get_dataloader(self, dataset_params = None, split = 'train', batch_size = 1, dataloader_params = None, drop_last = True):
        """
        Get the dataloader from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset;
        
        batch_size: int, optional, default: None. If batch_size is None, then the batch size parameter in the self.params will be used to represent the batch size (If still not specified, default: 4);
        
        dataloader_params: dict, optional, default: None. If dataloader_params is provided, then use the parameters specified in the dataloader_params to get the dataloader. Otherwise, the dataloader parameters in the self.params will be used to get the dataloader.

        Returns
        -------
        
        A torch.utils.data.DataLoader item.
        """
        from torch.utils.data import DataLoader

        # if split != "train":
        #     drop_last = True
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(dataset_params, split)
        if dataset is None:
            return None
        sampler = self.get_sampler(dataset, split, drop_last=drop_last)

        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_params
        )


def get_optimizer(model, optimizer_params = None, resume = False, resume_lr = None):
    """
    Get the optimizer from configuration.
    
    Parameters
    ----------
    
    model: a torch.nn.Module object, the model.
    
    optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer;
    
    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_lr: float, optional, default: None, the resume learning rate.
    
    Returns
    -------
    
    An optimizer for the given model.
    """
    from torch.optim import SGD, Adam, AdamW
    type = optimizer_params.get('type', 'AdamW')
    params = optimizer_params.get('params', {})

    if resume:
        network_params = [{'params': model.parameters(), 'initial_lr': resume_lr}]
        params.update(lr = resume_lr)
    else:
        network_params = model.parameters()
    if type == 'SGD':
        optimizer = SGD(network_params, **params)
    elif type == 'Adam':
        optimizer = Adam(network_params, **params)
    elif type == 'AdamW':
        optimizer = AdamW(network_params, **params)
    else:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizer

def get_lr_scheduler(optimizer, lr_scheduler_params = None, resume = False, resume_epoch = None, total_steps = None):
    """
    Get the learning rate scheduler from configuration.
    
    Parameters
    ----------
    
    optimizer: an optimizer;
    
    lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler;

    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_epoch: int, optional, default: None, the epoch of the checkpoint.
    
    Returns
    -------

    A learning rate scheduler for the given optimizer.
    """
    from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, StepLR, OneCycleLR
    type = lr_scheduler_params.get('type', '')
    params = lr_scheduler_params.get('params', {})
    if resume:
        params.update(last_epoch = resume_epoch)
    if type == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **params)
    elif type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, **params)
    elif type == 'CyclicLR':
        scheduler = CyclicLR(optimizer, **params)
    elif type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **params)
    elif type == 'StepLR':
        scheduler = StepLR(optimizer, **params)
    elif type == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, total_steps=total_steps, **params)
    elif type == '':
        scheduler = None
    else:
        raise NotImplementedError('Invalid learning rate scheduler type.')
    return scheduler
