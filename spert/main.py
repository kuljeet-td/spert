from spert.args import train_argparser, eval_argparser
from spert.config_reader import process_configs
from spert.spert import input_reader
import pandas as pd
from spert.spert.spert_trainer import SpERTTrainer

args_ = {'dataset_path': 'spert/data/datasets/conll04/conll04_prediction_example.json',
         'predictions_path': 'spert/data/predictions.json', 'spacy_model': 'en_core_web_sm',
         'config': 'spert/configs/example_predict.conf',
         'types_path': 'spert/data/datasets/conll04/spert_self_types.json',
         'tokenizer_path': 'spert/data/models/final_model', 'max_span_size': 10, 'lowercase': False,
         'sampling_processes': 4, 'model_path': 'spert/data/models/final_model', 'model_type': 'spert', 'cpu': False,
         'eval_batch_size': 1, 'max_pairs': 1000, 'rel_filter_threshold': 0.4, 'size_embedding': 25, 'prop_drop': 0.1,
         'freeze_transformer': False, 'no_overlapping': False, 'seed': None, 'cache_path': None, 'debug': False}


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


def remove_similar_substring(lst):
    if len(lst) > 1:
        sorted_lst = sorted(lst, key=lambda x: len(x.split(" ")))
        max_len = sorted_lst[-1].split(" ").__len__()
        for index in range(0, len(sorted_lst) - 1):
            if len(sorted_lst[index].split(" ")) < max_len:
                if sorted_lst[index] in ' '.join(sorted_lst[index + 1:]):
                    sorted_lst[index] = None
        return [i for i in sorted_lst if i]
    else:
        return lst


def make_list_keyphrases(out_pred_):
    df1 = pd.DataFrame(out_pred_)
    df1['indices'] = df1['entities'].apply(lambda a: [(_['start'], _['end']) for _ in a if
                                                      'SKILL' in _['type'] or 'jobfunction' in _[
                                                          'type'] or 'jobtitle' in _['type']])
    keyphrases_ = []
    for i, j in zip(df1['tokens'], df1['indices']):
        if j:
            ind_ = [i[x[0]:x[1]] for x in j]
        else:
            ind_ = []
        keyphrases_.extend(ind_)

    return remove_similar_substring([' '.join(keys) for keys in keyphrases_])


def __predict(predict_string):
    run_args = RunArgsBase(args_)
    trainer = SpERTTrainer(run_args)
    out_pred_ = trainer.predict(predict_string, dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                                input_reader_cls=input_reader.JsonPredictionInputReader)
    return make_list_keyphrases(out_pred_), out_pred_
