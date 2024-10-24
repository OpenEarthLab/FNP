import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger



def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("train", args.rundir, utils.get_rank(), filename='iter.log')
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    # build model
    logger.info('Building models ...')
    model = builder.get_model()
    model.kernel = utils.DistributedParallel_Model(model.kernel, args.local_rank)

    # build forecast model 
    logger.info('Building forecast models ...')
    args.forecast_model = builder.get_forecast(args.local_rank)

    # build dataset
    logger.info('Building dataloaders ...')
    dataset_params = args.cfg_params['dataset']
    train_dataloader = builder.get_dataloader(dataset_params=dataset_params, split='train', batch_size=args.batch_size)
    valid_dataloader = builder.get_dataloader(dataset_params=dataset_params, split='valid', batch_size=args.batch_size)
    # logger.info(f'dataloader length {len(train_dataloader), len(valid_dataloader)}')

    # train
    logger.info('begin training ...')
    model.train(train_dataloader, valid_dataloader, logger, args)
    logger.info('training end ...')


def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)
    
    args.cfg = os.path.join(args.rundir, 'training_options.yaml')
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

    cfg_params['dataloader']['num_workers'] = args.per_cpus
    cfg_params['dataset']['train']['length'] = args.lead_time // 6 + 2
    cfg_params['dataset']['valid']['length'] = args.lead_time // 6 + 2
    args.cfg_params = cfg_params
    
    args.rundir = os.path.join(args.rundir, f'mask{args.ratio}_lead{args.lead_time}h_res{args.resolution}')
    os.makedirs(args.rundir, exist_ok=True)

    if args.rank == 0:
        with open(os.path.join(args.rundir, 'train.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            # yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    subprocess_fn(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',           type = int,     default = 0,                        help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                        help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 4,                        help = 'number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 4,                        help = 'number of perCPUs to use')
    parser.add_argument('--max_epoch',      type = int,     default = 20,                       help = "maximum training epochs")
    parser.add_argument('--batch_size',     type = int,     default = 1,                        help = "batch size")
    parser.add_argument('--lead_time',      type = int,     default = 24,                       help = "lead time (h) for background")
    parser.add_argument('--ratio',          type = float,   default = 0.9,                      help = "mask ratio")
    parser.add_argument('--resolution',     type = int,     default = 128,                      help = "observation resolution")
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:19111',  help = 'multi process init method')
    parser.add_argument('--rundir',         type = str,     default = './configs/FNP',          help = 'where to save the results')

    args = parser.parse_args()

    main(args)

