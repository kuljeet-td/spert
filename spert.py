import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_trainer import SpERTTrainer


class RunArgsBase(object):
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


# def _predict():
#     arg_parser = predict_argparser()
#     process_configs(target=__predict)
#

def __predict():
    args_ = {'dataset_path': 'data/datasets/conll04/conll04_prediction_example.json',
            'predictions_path': 'data/predictions.json', 'spacy_model': 'en_core_web_sm',
            'config': 'configs/example_predict.conf', 'types_path': 'data/datasets/conll04/conll04_types.json',
            'tokenizer_path': 'data/models/conll04', 'max_span_size': 10, 'lowercase': False, 'sampling_processes': 4,
            'model_path': 'data/models/conll04', 'model_type': 'spert', 'cpu': False, 'eval_batch_size': 1,
            'max_pairs': 1000, 'rel_filter_threshold': 0.4, 'size_embedding': 25, 'prop_drop': 0.1,
            'freeze_transformer': False, 'no_overlapping': False, 'seed': None, 'cache_path': None, 'debug': False}
    run_args = RunArgsBase(args_)
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)


if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser(add_help=False)
    # arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    # args, _ = arg_parser.parse_known_args()
    __predict()
