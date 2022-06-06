import argparse
from logging import getLogger
import os
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.model.sequential_recommender.mbht import MBHT
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='retail', help='Benchmarks for session-based rec.')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        # 'neg_sampling': {'uniform':1},
        'neg_sampling': None,
        'benchmark_filename': ['train', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [5, 10, 101],
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'valid_metric': 'NDCG@10',
        'eval_args':{
            'mode':'full',
            'order':'TO'
            },
        'gpu_id':args.gpu_id,
        "MAX_ITEM_LIST_LENGTH":200,
        "train_batch_size": 32 if args.dataset == "ijcai_beh" else 64,
        "eval_batch_size":24 if args.dataset == "ijcai_beh" else 128,
        "hyper_len":10 if args.dataset == "ijcai_beh" else 6,
        "scales":[10, 4, 20],
        "enable_hg":1,
        "enable_ms":1,
        "customized_eval":1,
        "abaltion":""
    }
        
    config = Config(model="MBHT", dataset=f'{args.dataset}', config_dict=config_dict)
    # config['device']="cpu"
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, log_root="log")
    logger = getLogger()

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    train_sampler, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
        train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    else:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config['show_progress']
    )

    logger.info(set_color('test result', 'yellow') + f': {test_result}')
