import argparse, os, torch, tqdm, time, inspect, sys, ast, itertools, math, gc, re, glob, pickle
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from accelerate import Accelerator
from evaluate import load
from torch.optim import AdamW
import torch.mps as mps
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import namedtuple


class TrainingConfig:
    """
    This is the configuration object which will be used almost everywhere. It may become global
    for convenience but not encouraged.
    """

    def __init__(self):
        self.working_dir = None
        self.training_record = "training.csv"  # To save training details for each epoch
        self.latest_weight = "latest_weight.pth"
        self.freq, self.prediction_length = "1M", 24
        self.time_features = [month_of_year]
        self.model_cfg: TimeSeriesTransformerConfig = None  # Initialised by training_prepare
        self.train_dataset, self.test_dataset, self.val_dataset = None, None, None
        self.epochs = None  # Initialised by training policies
        self.batch_sizes = {'train': 32, 'test': 16}  # Initialised by training policies
        self.lr_init = None  # Initialised by training policies
        self.data = None  # Initialised by training_prepare
        self.device = None  # Initialised by training_prepare
        self.lags_sequence = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
        self.model = None  # Initialised by training_prepare
        self.optimizer = None  # Initialised by training_prepare
        self.lr_scheduler = None
        self.lr_gamma = None
        self.lr_step = None
        self.train_dataloader = None  # Initialised by training_prepare
        self.test_dataloader = None  # Initialised by training_prepare
        self.accelerator = None  # Initialised by training_prepare
        self.last_epoch = None
        self.current_epoch = None
        self.log_lines = []

    def update_from(self, args: argparse.Namespace):
        for key, value in vars(args).items():
            setattr(self, key, value)

    def update_working_path(self):
        this_file = os.path.abspath(inspect.getsourcefile(lambda: 0))
        if os.path.islink(this_file):
            self.working_dir = os.path.dirname(os.readlink(this_file))
        else:
            self.working_dir = os.path.dirname(this_file)
        [os.makedirs(os.path.join(self.working_dir, d), exist_ok=True)
         for d in [self.save_dir, self.results_dir]]

    def print_and_log(self, string: str, **kwargs):
        print(string, **kwargs)
        self.log_lines.append(string) if string not in self.log_lines else ()  # 防止重复

    def __repr__(self):
        """
        Prepare a string containing import information of this class
        :return: the string representing  class Configurations
        """
        string_list = [f"",
                       f"",
                       f"",
                       f"", ]
        return '\n'.join(string_list)


# region ✅ 1 Data preparation functions
class TrainingData:
    def __init__(self, cfg: TrainingConfig):
        """
        Assume there is a directory named 'dataset' under current directory and there are 3 csv files
        named 'tourism-train.csv', 'tourism-val.csv', 'tourism-test.csv'
        :param cfg: the configuration class of this program
        """
        self.training_config = cfg
        try:
            file_list = ['tourism-train.csv', 'tourism-val.csv', 'tourism-test.csv']
            dataframes = [pd.read_csv(os.path.join(cfg.working_dir, 'dataset', f),
                                      converters={"target": parse_list, "feat_static_cat": parse_list},
                                      parse_dates=['start'])
                          for f in file_list]
            df_train, df_val, df_test = dataframes
            self.df3 = {'train': df_train, 'validation': df_val, 'test': df_test}
        except Exception as e:
            print_exception(e)
            panic("Loading dataset failed ...")

    def __getitem__(self, key):
        return self.df3[key]

    def check_data_length(self):
        for idx in range(len(self.df3['train'])):
            assert len(self.df3['train'].loc[idx, 'target']) + 24 == len(self.df3['validation'].loc[idx, 'target'])

    @staticmethod
    def time_features_transform(start_data, target_data, time_features, prediction_length, is_train: bool,
                                dtype=np.float32):
        """
        # https://ts.gluon.ai/stable/_modules/gluonts/transform/feature.html#AddTimeFeatures
        """
        length = target_data.shape[0] if is_train else target_data.shape[0] + prediction_length
        index = pd.period_range(start_data, periods=length, freq=start_data.freq)
        data = np.vstack([feat(index) for feat in time_features]).astype(dtype)
        return data

    @staticmethod
    def age_features_transform(target_data, prediction_length, is_train: bool,
                               log_scale: bool = True, dtype=np.float32):
        length = target_data.shape[0] if is_train else target_data.shape[0] + prediction_length
        if log_scale:
            age = np.log10(2.0 + np.arange(length, dtype=dtype))
        else:
            age = np.arange(length, dtype=dtype)
        target_data = age.reshape((1, length))
        return target_data

    def __t_remove_unused_columns(self):
        cfg = self.training_config
        model_cfg = cfg.model_cfg
        remove_field_names = []
        remove_field_names.append("feat_static_real") if model_cfg.num_static_real_features == 0 else ()
        remove_field_names.append("feat_dynamic_real") if model_cfg.num_dynamic_real_features == 0 else ()
        remove_field_names.append("feat_static_cat") if model_cfg.num_static_categorical_features == 0 else ()
        for k, df in self.df3.items():  # Step 2, Remove unused columns
            columns_to_remove = [c for c in remove_field_names if c in df.columns]
            self.df3[k].drop(columns=columns_to_remove, inplace=True)

    def __t_start_target_and_features(self):
        cfg = self.training_config
        model_cfg = cfg.model_cfg
        for k, df in self.df3.items():
            new_column = []
            for idx in range(len(df)):
                df.loc[idx, 'start'] = pd.Period(df.loc[idx, 'start'], cfg.freq)
                values = np.array(df.loc[idx, 'target'], dtype=np.float32)
                new_column.append(values)
                if model_cfg.num_static_categorical_features > 0:
                    df.loc[idx, 'feat_static_cat'] = np.asarray(df.loc[idx, 'feat_static_cat'], dtype=int)
                if model_cfg.num_static_real_features > 0:
                    df.loc[idx, 'feat_static_real'] = np.asarray(df.loc[idx, 'feat_static_real'], dtype=int)
            df['new_target'] = new_column

    def __t_add_observed_values(self):
        for k, df in self.df3.items():
            new_column = []
            for idx in range(len(df)):
                value = np.array(df.loc[idx, 'target'], dtype=np.float32)
                nan_entries = np.isnan(value)
                if nan_entries.any():
                    value = value.copy()
                    print("\033[31mNaN values in __t_add_observed_values")
                observed_mask = np.invert(nan_entries, out=nan_entries).astype(value.dtype, copy=True)
                new_column.append(observed_mask)
            df['observed_mask'] = new_column

    def __t_add_time_features(self):
        cfg = self.training_config
        model_cfg = cfg.model_cfg
        length = model_cfg.prediction_length
        for k, df in self.df3.items():
            new_column = []
            is_train = True if k in ['train'] else False
            for idx in range(len(df)):
                start, target = df.loc[idx, 'start'], np.array(df.loc[idx, 'target'])
                features_1 = self.time_features_transform(start, target, cfg.time_features, length, is_train=is_train)
                features_2 = self.age_features_transform(target, length, is_train=is_train)
                time_features = np.vstack([features_1, features_2])
                new_column.append(time_features)
            df['time_features'] = new_column
        return

    def do_transforms(self):
        self.__t_remove_unused_columns()
        self.__t_start_target_and_features()
        self.__t_add_observed_values()
        self.__t_add_time_features()
        for k, df in self.df3.items():
            df = df[['start', 'item_id', 'feat_static_cat', 'time_features', 'new_target', 'observed_mask']]
            self.df3[k] = df.rename(columns={"new_target": "values", "feat_static_cat": "static_categorical_features"})


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pd_data: pd.DataFrame, transforms=None):
        super(torch.utils.data.Dataset).__init__()
        self.transforms = transforms
        self.data = pd_data

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.data.shape[0]:
            x = self.data.loc[self.n]
            self.n += 1
            if self.transforms:
                for t in self.transforms:
                    x = t(x)
            return x
        else:
            raise StopIteration

    def __getitem__(self, index):
        x = self.data.loc[index]
        if self.transforms:
            for t in self.transforms:
                x = t(x)
        return x

    def get_column(self, column: str):
        try:
            return self.data.loc[:, column].tolist()
        except Exception as e:
            print_exception(e)
            return None

    def __repr__(self):
        return str(self.data)


class MySampler:
    def __init__(self, is_train: bool):
        self.n: int = 0
        self.num_instances: float = 1  # Always 1 for single process
        self.total_length: int = 0
        self.is_train = is_train

    def __call__(self, *args, **kwargs):
        return self.train_sample(*args, **kwargs) if self.is_train else self.prediction_sample(*args, **kwargs)

    def train_sample(self, ts: np.ndarray, prediction_length) -> np.ndarray:
        window_size = ts.shape[0] - prediction_length + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n
        if avg_length <= 0:
            return np.array([], dtype=int)
        p = self.num_instances / avg_length
        (indices,) = np.where(np.random.random_sample(window_size) < p)
        return indices

    @staticmethod
    def prediction_sample(ts: np.ndarray, prediction_length) -> np.ndarray:
        # We create a Test Instance splitter which will sample the very last @gluonts-transformer-gluonts.py
        b = ts.shape[0]
        return np.array([b])


class MyInstanceSplitter:
    def __init__(self, sampler, past_length, future_length, lead_time=0):
        self.sampler = sampler
        self.future_length = future_length  # in this case, 24
        self.past_length = past_length  # in this case, 85
        self.lead_time = lead_time
        self.dummy_value = 0.0
        self.dtype = np.float32
        # output_NTC whether to have time series output in (time, dimension) or in (dimension, time)
        # layout (default: True)
        _ = ("https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?"
             "highlight=instancesplitter#gluonts.transform.InstanceSplitter")
        self.output_NTC = True

    def __call__(self, *args, **kwargs):
        return self.split(*args, **kwargs)

    def split(self, target: np.ndarray, time_features: np.ndarray, observed_mask: np.ndarray):
        pl, lt = self.future_length, self.lead_time
        sampled_indices = self.sampler(target, self.future_length)
        past_pieces, future_pieces = [], []
        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            # Copy 3 arrays to do some jobs
            target_cp, time_features_cp, observed_mask_cp = target.copy(), time_features.copy(), observed_mask.copy()
            past_tuple, future_tuple = [], []
            for ts_field in [target_cp, time_features_cp, observed_mask_cp]:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = ts_field[..., i - self.past_length: i]
                elif i < self.past_length:
                    pad_block = np.ones(ts_field.shape[:-1] + (pad_length,), dtype=self.dtype) * self.dummy_value
                    past_piece = np.concatenate([pad_block, ts_field[..., :i]], axis=-1)
                else:
                    past_piece = ts_field[..., :i]
                future_piece = ts_field[..., i + lt: i + lt + pl]
                if self.output_NTC:
                    past_piece = past_piece.transpose()
                    future_piece = future_piece.transpose()
                past_tuple.append(past_piece)
                future_tuple.append(future_piece)
            pad_indicator = np.zeros(self.past_length, dtype=target.dtype)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1
            past_pieces.append(past_tuple)
            future_pieces.append(future_tuple)
        return past_pieces, future_pieces


class MyDataLoader:
    def __init__(self, dataset: MyDataset, model_cfg, batch_size, num_batches_per_epoch, is_train):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        epochs_to_iter_all_data = math.ceil(len(self.dataset) / batch_size)  # 多少个epoch就可以遍历所有数据
        self.length = num_batches_per_epoch if is_train else min(epochs_to_iter_all_data, num_batches_per_epoch)
        self.is_train = is_train
        self.ext_dataset_len_range = range(max(len(self.dataset), batch_size * num_batches_per_epoch))
        self.index_iter = itertools.cycle(range(len(self.dataset)) if is_train else self.ext_dataset_len_range)
        train_sampler, test_sampler = MySampler(is_train=True), MySampler(is_train=False)
        future_length = model_cfg.prediction_length
        past_length = model_cfg.context_length + max(model_cfg.lags_sequence)
        train_spliter = MyInstanceSplitter(train_sampler, past_length, future_length)
        test_spliter = MyInstanceSplitter(test_sampler, past_length, future_length)
        self.split_function = train_spliter if is_train else test_spliter
        self.old_batch_pieces = None
        if not is_train:
            assert batch_size * num_batches_per_epoch > len(self.dataset)

    def __reset_iter(self):
        self.index_iter = itertools.cycle(range(len(self.dataset)) if self.is_train else self.ext_dataset_len_range)

    def __iter__(self):
        self.n = 0
        return self

    def __row_to_batch(self, row, old_batch_pieces=None):
        new_batch_pieces = None
        l_past_time_features, l_past_values, l_past_observed_masks = [], [], []
        l_future_time_features, l_future_values, l_future_observed_masks = [], [], []
        l_static_categorical_features = []
        target, time_features, observed_mask = row['values'], row['time_features'], row['observed_mask']
        returned_values = self.split_function(target, time_features, observed_mask)
        if returned_values and returned_values[0] and returned_values[1]:
            past_pieces, future_pieces = returned_values
            assert len(past_pieces) == len(future_pieces)
            for i in range(len(past_pieces)):
                past_value, past_time_feature, past_observed_mask = past_pieces[i]
                future_value, future_time_feature, future_observed_mask = future_pieces[i]
                l_past_values.append(torch.tensor(past_value))
                l_past_time_features.append(torch.tensor(past_time_feature))
                l_past_observed_masks.append(torch.tensor(past_observed_mask))
                l_future_values.append(torch.tensor(future_value))
                l_future_time_features.append(torch.tensor(future_time_feature))
                l_future_observed_masks.append(torch.tensor(future_observed_mask))
                l_static_categorical_features.append(torch.tensor(row['static_categorical_features']))
            l_past_time_features = torch.stack(l_past_time_features, dim=0)
            l_past_values = torch.stack(l_past_values, dim=0)
            l_past_observed_masks = torch.stack(l_past_observed_masks, dim=0)
            l_future_time_features = torch.stack(l_future_time_features, dim=0)
            l_future_values = torch.stack(l_future_values, dim=0)
            l_future_observed_masks = torch.stack(l_future_observed_masks, dim=0)
            l_static_categorical_features = torch.stack(l_static_categorical_features, dim=0)
            new_batch_pieces = {'past_time_features': l_past_time_features, 'past_values': l_past_values,
                                'past_observed_mask': l_past_observed_masks,
                                'future_time_features': l_future_time_features, 'future_values': l_future_values,
                                'future_observed_mask': l_future_observed_masks,
                                'static_categorical_features': l_static_categorical_features}
        if new_batch_pieces is not None and next(iter(new_batch_pieces.values())).shape[0] > 0:
            if old_batch_pieces is not None and next(iter(old_batch_pieces.values())).shape[0] > 0:
                for key in old_batch_pieces.keys():
                    new_batch_pieces[key] = torch.cat((old_batch_pieces[key], new_batch_pieces[key]), dim=0)
        else:
            new_batch_pieces = old_batch_pieces
        return new_batch_pieces

    def __next__(self):
        if self.n < self.length:
            batch = self.old_batch_pieces
            while True:  # 32x10=320, 32x11=352, 32x12 = 384 > 366
                try:
                    index = next(self.index_iter)
                    idx_error = index >= len(self.dataset)
                    if not idx_error:
                        batch = self.__row_to_batch(self.dataset[index], batch)
                    else:
                        self.__reset_iter()
                    if (batch and next(iter(batch.values())).shape[0] >= self.batch_size) or idx_error:
                        self.n += 1
                        if idx_error:
                            assert self.n == self.length
                        batch_data = {k: v[:self.batch_size] for k, v in batch.items()}
                        self.old_batch_pieces = {k: v[self.batch_size:] for k, v in batch.items()}
                        return batch_data
                except Exception as e:
                    panic(str(e))
        else:
            raise StopIteration

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch = self.__row_to_batch(self.dataset[idx])
        return batch


# endregion


# region ✅ 2 Model preparation functions
def prepare_model(cfg: TrainingConfig):
    model_config = TimeSeriesTransformerConfig(
        prediction_length=cfg.prediction_length,
        # context length:
        context_length=cfg.prediction_length * 2,
        # lags coming from helper given the freq:
        lags_sequence=cfg.lags_sequence,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(cfg.time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=1,
        # it has 366 possible values:
        cardinality=[len(cfg.data.df3['train'])],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2],
        # transformer params: ⚠️decoder_layers more important
        encoder_attention_heads=cfg.attention_heads,
        decoder_attention_heads=cfg.attention_heads,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        d_model=128, )  # d_model – the number of expected features in the input，default was 32
    model = TimeSeriesTransformerForPrediction(model_config)
    if cfg.save_model:
        model_txt_lines = str(model).splitlines()
        model_file = 'hf-TimeSeries-Transformer.txt'.lower()
        model_file = os.path.join(cfg.working_dir, cfg.save_dir, model_file.lower())
        save_to_file(model_file, model_txt_lines)
        model_config_txt_lines = str(model_config).splitlines()
        cfg_file = 'hf-TimeSeries-Transformer-cfg.txt'
        cfg_file = os.path.join(cfg.working_dir, cfg.save_dir, cfg_file.lower())
        save_to_file(cfg_file, model_config_txt_lines)
        print(f"Model saved to        \033[32m{model_file}\033[0m\n" +
              f"Model config saved to \033[32m{cfg_file}\033[0m")
    return model, model_config


# endregion


# region ✅ 3 Train supporting functions
def calc_lr_decay(lr_init: float, lr_100: float, decay_step: float = 1) -> tuple[float, float]:
    steps = 100 / decay_step
    return math.exp(math.log(lr_100 / lr_init) / steps), decay_step


def month_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as value between [-0.5, 0.5]
    """
    return (index.month.values - 1) / 11.0 - 0.5


def calc_mase_and_smape(cfg: TrainingConfig, forecasts, dataset: MyDataset):
    forecasts = forecasts[:len(dataset), :, :]
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")
    forecast_median = np.median(forecasts, axis=1)
    mase_metrics, smape_metrics = [], []
    test_dataset = dataset.get_column('values')
    for item_id, ts in enumerate(test_dataset):
        training_data = ts[:-cfg.prediction_length]
        ground_truth = ts[-cfg.prediction_length:]
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=12)  # We knew that get_seasonality("1M")=12
        mase_metrics.append(mase["mase"])

        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
        )
        smape_metrics.append(smape["smape"])
    mase_value, smape_value = np.mean(mase_metrics), np.mean(smape_metrics)
    return mase_value, smape_value


def empty_gpu_cache(cfg: TrainingConfig):
    gc.collect()
    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()
    elif cfg.device.type == 'mps':
        mps.empty_cache()
    else:
        return


def gpu_mem_in_use(cfg: TrainingConfig) -> str:  # returns GB of gpu mem left in str format
    if cfg.device.type == 'cuda':
        max_memory_allocated = torch.cuda.max_memory_allocated()
    elif cfg.device.type == 'mps':
        max_memory_allocated = mps.current_allocated_memory()
    else:
        max_memory_allocated = 0.0
    return f"{max_memory_allocated / 1024 ** 3:.2f}GB"


def match_layer_name(inputs: str):
    try:
        m = re.search(r"(model\.(encoder|decoder)\.layers\.\d+)\.", inputs)
        return m[1]
    except Exception as e:  # We don't care exception here
        return None


def load_weights_file(cfg: TrainingConfig, weights_file: str, adaptive: bool = False):
    model = cfg.model
    try:
        weights = torch.load(weights_file, map_location=cfg.device)
        all_keys = list(weights.keys())
        all_encoder_keys = [match_layer_name(layer) for layer in all_keys if layer.startswith('model.encoder.layers')]
        all_decoder_keys = [match_layer_name(layer) for layer in all_keys if layer.startswith('model.decoder.layers')]
        all_encoder_keys = remove_duplicates(all_encoder_keys)
        all_decoder_keys = remove_duplicates(all_decoder_keys)
        encoder_layers, decoder_layers = len(all_encoder_keys), len(all_decoder_keys)
        if adaptive:
            print("\033[1;35mModel layers adjusted according to weights file\n\033[0m"
                  f"encoder layers from {cfg.encoder_layers} to \033[1;36m{encoder_layers}\033[0m\n"
                  f"decoder layers from {cfg.decoder_layers} to \033[1;36m{decoder_layers}\033[0m")
            cfg.encoder_layers, cfg.decoder_layers = encoder_layers, decoder_layers
            model, model_cfg = prepare_model(cfg)
            model = model.to(cfg.device)
            cfg.model, cfg.model_cfg = model, model_cfg
        if cfg.debug:
            print(f"{all_encoder_keys}\n{all_decoder_keys}")
            sys.exit(0)
        model.load_state_dict(weights)
        print(f"\033[1;35mLoading {weights_file} into model...\033[0m")
        return model
    except Exception as e:
        print(f"Exception = {str(e)}\ndevice = {cfg.device}")
        panic(f"\033[1;31m{weights_file} does not match model, exiting...")


def get_current_lr(cfg: TrainingConfig) -> float:
    return cfg.optimizer.param_groups[-1]['lr']


def ansi_only_str(line: str) -> str:
    ansi_escapes = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return re.sub(ansi_escapes, '', line)


def get_file_contents_v4(file_name: str, remove_comments: bool = False, ansi_only: bool = False) -> [str]:
    """
    :param file_name: The file to read as text file
    :param remove_comments: comments with '#' will be removed if set
    :param ansi_only: remove non ansi characters which are most likely colour codes
    :return: a list of str
    """
    try:
        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f]  # remove left and right white spaces and '\n'
            lines = [ansi_only_str(line).strip() for line in lines] if ansi_only else lines
            lines = [re.sub("#.*", '', line).strip() for line in lines] if remove_comments else lines
            lines = [line for line in lines if line]  # exclude empty lines
            return lines
    except Exception as e:
        print_exception(e)
        return []


def get_old_csv_lines(cfg: TrainingConfig, include_header=False, include_records=True, ansi_only=False) -> list[str]:
    try:
        lines_with_ansi = get_file_contents_v4(os.path.join(cfg.working_dir, cfg.training_record))
        lines_ansi_only = [ansi_only_str(line) for line in lines_with_ansi]
        new_contents = []
        indices = [idx for idx, string in enumerate(lines_ansi_only) if 'epoch' in string.lower()]
        first_record_idx = indices[0] if indices else None
        all_lines = lines_with_ansi if ansi_only else lines_ansi_only
        header = all_lines[:first_record_idx] if first_record_idx else []
        records = all_lines[first_record_idx:] if first_record_idx else []
        if include_header:
            new_contents += header
        if include_records:
            new_contents += records
        return new_contents
    except Exception as e:
        print_exception(e)
        return []


def get_details_from_record_line(a_line: str):
    """
    08-22 08:43 4.23GB Epoch 130/300 loss=5.6039 MASE:   1.6554 SMAPE:   0.1973 lr=1.8400e-04
    :param a_line:
    :return: a tuple of five values
    """
    a_line = ansi_only_str(a_line)
    RecordInfo = namedtuple('RecordInfo', ['time', 'epoch', 'loss', 'mase', 'smape', 'lr'])
    try:
        m_time = re.search(r"(\d\d-\d\d \d\d:\d\d)", a_line)
        epoch_time = m_time[1]
        m_epoch = re.search(r"Epoch.*?(\d+)/", a_line, flags=re.IGNORECASE)
        epoch = int(m_epoch[1])
        m_loss = re.search(r"loss=(\d+\.\d+)", a_line, flags=re.IGNORECASE)
        loss = float(m_loss[1])
        try:
            m_mase = re.search(r"MASE:\s*(\d+\.\d+)", a_line, flags=re.IGNORECASE)
            mase = float(m_mase[1])
            m_smape = re.search(r"MASE:\s*(\d+\.\d+)", a_line, flags=re.IGNORECASE)
            smape = float(m_smape[1])
        except Exception as e:  # mase & smape can be absent
            mase, smape = None, None
        lr = float(a_line.split('=')[-1])
        return RecordInfo(epoch_time, epoch, loss, mase, smape, lr)
    except Exception as e:
        print_exception(e)
        return None


def get_last_epoch(record_lines) -> int:
    last_line = record_lines[-1]
    return get_details_from_record_line(last_line).epoch


def get_last_lr(record_lines) -> float:
    last_line = record_lines[-1]
    return get_details_from_record_line(last_line).lr


def process_resume(cfg) -> list[str]:
    existing_records = []
    if cfg.resume:
        existing_records = get_old_csv_lines(cfg, include_header=False, include_records=True, ansi_only=False)
        weights_file = cfg.weights if cfg.weights else os.path.join(cfg.working_dir, cfg.results_dir, cfg.latest_weight)
        model = load_weights_file(cfg, weights_file)
        cfg.last_epoch = get_last_epoch(existing_records)
        lr_resumed = cfg.lr if cfg.lr else get_last_lr(existing_records)
        cfg.optimizer = AdamW(cfg.model.parameters(), lr=lr_resumed, betas=(0.9, 0.95), weight_decay=1e-1)
        cfg.lr_scheduler = lr_scheduler.StepLR(cfg.optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
        if cfg.lr_step == 1:
            cfg.lr_scheduler.step()
        print(f"\033[35mContinuing from \033[1;32m Epoch {cfg.last_epoch}\033[0m")
    else:
        model = cfg.model
        results_dir = os.path.join(cfg.working_dir, cfg.results_dir)
        [os.remove(f) for f in glob.glob(results_dir.rstrip('/') + '/*.pth')]
        [os.remove(f) for f in glob.glob(results_dir.rstrip('/') + '/*.dat')]
    record_file = os.path.join(cfg.working_dir, cfg.training_record)
    os.remove(record_file) if os.path.isfile(record_file) else ()  # prepare_log会重建这个文件
    return model, existing_records


def prepare_log(cfg: TrainingConfig, existing_records: list[str]):
    training_header = "\033[1;36mStub training log header, will be prepared later\033[0m"
    [log_to_file(cfg, line) for line in cfg.log_lines]
    log_to_file(cfg, training_header)
    if cfg.resume and cfg.last_epoch:
        [log_to_file(cfg, line) for line in existing_records] if existing_records else ()


def loss_and_epoch_from_file_name(file_name: str):  # epoch-010-6.3011-.pth
    Loss_Epoch = namedtuple('Loss_Epoch', ['loss', 'epoch'])
    try:
        m_loss = re.search(r"epoch-\d+-(\d+\.\d+)", file_name)
        m_epoch = re.search(r"epoch-(\d+)-", file_name)
        return Loss_Epoch(float(m_loss[1]), m_epoch[1])
    except Exception as e:
        print_exception(e)
        return None


def epoch_from_forecast_file_name(file_name: str) -> str:
    try:  # forecast-003.dat
        m = re.search(r"-(\d+)\.dat", file_name)
        return m[1]
    except Exception as e:
        print_exception(e)
        return None


def keep_best_weight_files(cfg: TrainingConfig, current_mase, top_n: int):
    results_dir = os.path.join(cfg.working_dir, cfg.results_dir)
    pth_files = glob.glob(results_dir.rstrip('/') + '/*.pth')
    pth_files = [file for file in pth_files if cfg.latest_weight not in file]
    # loss_values = [loss_from_file_name(file) for file in pth_files]
    pth_files.sort(key=lambda file: loss_and_epoch_from_file_name(file).loss)
    pth_files_to_delete = pth_files[top_n:]  # files with larger loss values
    if current_mase and cfg.mase_limit and current_mase > cfg.mase_limit:  #
        file_name_partial = f"epoch-{cfg.current_epoch:03d}"
        current_epoch_pth = [file for file in pth_files if file_name_partial in file][0]
        pth_files_to_delete.append(current_epoch_pth)
        f = os.path.basename(current_epoch_pth)
        print(f"\033[1;35m{f} mase = {current_mase:.4f} > {cfg.mase_limit} and will be dropped\033[0m")
    try:
        forecast_files = glob.glob(results_dir.rstrip('/') + '/*.dat')  # forecast-003.dat
        epochs_to_delete = [loss_and_epoch_from_file_name(f).epoch for f in pth_files_to_delete]
        forecast_files_to_delete = [f for f in forecast_files if epoch_from_forecast_file_name(f) in epochs_to_delete]
        [os.remove(f) for f in forecast_files_to_delete]
        [os.remove(f) for f in pth_files_to_delete]
    except Exception as e:
        print_exception(e)


def actions_after_epoch(cfg: TrainingConfig, bar_train: str, mase: float, smape: float):
    """
    Just logging, weights，and early_stop,
    08-21 14:18 3.30GB Epoch  29/300 loss=6.8250 MASE: ________, SMPAE ________ lr = 4.6422e-04
    """
    mase_str = f"MASE: {mase:8.4f}" if mase is not None else f"MASE: {'_' * 8}"
    smape_str = f"SMAPE: {smape:8.4f}" if smape is not None else f"SMAPE: {'_' * 8}"
    bar_train = ansi_only_str(bar_train)
    bar_train = re.sub(r"lr=.*?(loss)(.*?) EFT.*", r"\1\2", bar_train)
    bar_train = f"{bar_train} {mase_str} {smape_str} lr={get_current_lr(cfg):.4e} "
    log_to_file(cfg, bar_train)
    latest_weight = os.path.join(cfg.working_dir, cfg.results_dir, cfg.latest_weight)
    torch.save(cfg.model.state_dict(), latest_weight)
    keep_best_weight_files(cfg, current_mase=mase, top_n=cfg.top_n_files)
    if cfg.early_stop:  # Monitor training trend specified by args, in number of epochs
        lines_epoch = get_old_csv_lines(cfg, include_header=False, include_records=True, ansi_only=True)
        performance = epochs_loss_not_decreasing(lines_epoch)
        print(f"MASE: {mase:.4f}\tSMAPE: {smape:.4f}, BEST: {performance.best_loss:.4f}@{performance.best_epoch:3d} "
              f"NO Improvement epochs: {performance.epochs_no_improvement:3d}")
        if performance.epochs_no_improvement >= cfg.early_stop:
            print(f"\033[1;31mTraining stopped with {performance.epochs_no_improvement} epochs no improvement\033[0m\n"
                  f"\033[1;35mBest loss = {performance.best_loss} at epoch {performance.best_epoch}033[0m")
            sys.exit(0)
    else:
        print(f"MASE: {mase:.4f}\tSMAPE: {smape:.4f}")


def epochs_loss_not_decreasing(lines_epoch: str):
    TrainingTrend = namedtuple('TrainingTrend', ['epochs_no_improvement', 'best_epoch', 'best_loss'])
    assert len(lines_epoch) > 0
    loss_value = [get_details_from_record_line(line).loss for line in lines_epoch]
    assert len(loss_value) > 0
    loss_value_np = np.array(loss_value)
    loss_min_index = np.argsort(loss_value_np)[0]
    return TrainingTrend(loss_value_np.shape[0] - loss_min_index - 1, loss_min_index + 1, loss_value_np[loss_min_index])


# endregion


# region ✅ 4 Other supporting functions
def remove_duplicates(a_list, condition=lambda i, e, l: e not in l[:i]):
    return [element for idx, element in enumerate(a_list) if condition(idx, element, a_list)]


def print_exception(e: Exception):
    caller = inspect.currentframe().f_back.f_code.co_name
    print(f"❌\033[1;31mException from \033[35m{caller}\033[1;31m: {str(e)}\033[0m❌")


def safe_cast_float(string: str, default=0.0) -> float:
    try:
        return float(string)
    except (ValueError, TypeError):
        return default


def save_to_file(file_name: str, string_list: list[str] = None, append=False):
    f = open(file_name, 'a') if append else open(file_name, 'w')
    [f.write(line + '\n') for line in string_list]
    f.close()


def sep_line(symbol='#', repeats=100):
    print(f"\033[36m{symbol}\033[0m" * repeats)


def parse_list(column_value):
    try:
        return ast.literal_eval(column_value)
    except (ValueError, SyntaxError):
        return column_value


def debug_print(cfg, *argv, **kwargs):
    caller = inspect.currentframe().f_back.f_code.co_name
    print(f"🟡\033[35m{caller}:\033[0m🟡", *argv, **kwargs) if cfg.debug else ()


def panic(*argv):
    caller = inspect.currentframe().f_back.f_code.co_name
    print(f"🔴🔴\033[35m{caller}:\033[0m️🔴🔴", *argv, sep='\n')
    sys.exit(2)


def early_exit_by_ctrl_c(cfg: TrainingConfig):
    print("Ctrl +C was pressed ...")
    this_func = inspect.currentframe().f_code.co_name
    print(f"Add your actions in \033[35m{this_func}\033[0m before program exits...")
    sys.exit(0)


def tqdm_estimated_finish_time(t: tqdm.std.tqdm) -> str:
    rate = t.format_dict["rate"]
    remaining = max(t.total - t.n - 1, 0) / rate if rate and t.total else 0  # Seconds*
    eft = datetime.datetime.fromtimestamp(time.time() + remaining)
    today = datetime.datetime.now().day
    eft_str = eft.strftime("EFT %H:%M" if eft.day == today else "EFT %a %H:%M\033[0m")
    return eft_str


def plot_train_val_example(cfg):
    idx = np.random.randint(cfg.data['train'].shape[0])
    figure, axes = plt.subplots()
    axes.plot(cfg.data.df3['train'].loc[idx, "target"], color="blue")
    axes.plot(cfg.data.df3['validation'].loc[idx, "target"], color="red", alpha=0.2)
    plt.title(f"Series {idx} from train and validation", color='blue')
    plt.show()


def plot_forecast(cfg, forecasts, ts_index: int, dataset, title: str = None):
    fig, ax = plt.subplots()
    prediction_length = cfg.prediction_length
    index = pd.period_range(start=dataset[ts_index]['start'], periods=dataset[ts_index]['values'].shape[0],
                            freq=cfg.freq).to_timestamp()
    # Major ticks every half year, minor ticks every month,
    single_forecast = forecasts[ts_index] if forecasts.shape[0] > 1 else forecasts[0]
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.plot(index[-2 * prediction_length:], dataset[ts_index]['values'][-2 * prediction_length:], label="actual")
    plt.plot(index[-prediction_length:], np.median(single_forecast, axis=0), label="median")
    plt.fill_between(index[-prediction_length:], single_forecast.mean(0) - single_forecast.std(axis=0),
                     single_forecast.mean(0) + single_forecast.std(axis=0), alpha=0.3, interpolate=True,
                     label="+/- 1-std")
    plt.title(title, color='blue') if title else ()
    plt.legend()
    plt.show()


def print_data_base_info(cfg: TrainingConfig):
    keys = sorted(list(cfg.data.df3.keys()), key=lambda x: len(x))
    maxlen = len(keys[-1])
    [print(f"{k:{maxlen}} {len(v)} records, {list(v.columns)}") for k, v in cfg.data.df3.items()]


def formatted_summary(cfg: TrainingConfig):
    model, model_cfg = cfg.model, cfg.model_cfg
    str_model_cfg = str(model_cfg)
    print(f"{str(model)}\n{str_model_cfg}")


def get_model_key_params(cfg: TrainingConfig):
    model, model_cfg = cfg.model, cfg.model_cfg
    encoder_attention_heads = model_cfg.encoder_attention_heads
    decoder_attention_heads = model_cfg.decoder_attention_heads
    encoder_layers = model_cfg.encoder_layers
    decoder_layers = model_cfg.decoder_layers
    dropouts = [f"{k}->{getattr(model_cfg, k):.1f}" for k in dir(model_cfg) if 'dropout' in k]
    dropouts = ", ".join(dropouts)
    d_model = model_cfg.d_model
    output_str = (f"d_model={d_model}, encoder_layers={encoder_layers}, decoder_layers={decoder_layers}\n"
                  f"encoder_attention_heads={encoder_attention_heads} "
                  f"decoder_attention_heads={decoder_attention_heads}\n"
                  f"dropouts = {dropouts}")
    return output_str


def log_to_file(cfg: TrainingConfig, string: str = ""):
    file = open(os.path.join(cfg.working_dir, cfg.training_record), 'a')
    file.write(string + '\n')
    file.close()


def save_to_file(file, string_list=[], append=False):
    f = open(file, 'a') if append else open(file, 'w')
    [f.write(line + '\n') for line in string_list]
    f.close()


# endregion


# region ❤️ 5 Major functions
def training_policy_1(cfg: TrainingConfig, lr_init=None):
    cfg.epochs = 300
    cfg.lr_init = lr_init if lr_init is not None else 6e-4
    cfg.optimizer = AdamW(cfg.model.parameters(), lr=cfg.lr_init, betas=(0.9, 0.95), weight_decay=1e-1)
    cfg.lr_gamma, cfg.lr_step = calc_lr_decay(lr_init=cfg.lr_init, lr_100=cfg.lr_init * 0.4)
    cfg.lr_scheduler = lr_scheduler.StepLR(cfg.optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    lr_100 = cfg.lr_init * cfg.lr_gamma ** (100 / cfg.lr_step)
    cfg.print_and_log(f"Optimizer: {type(cfg.optimizer)}, lr={cfg.lr_init:.2e} lr_100={lr_100:.2e}")


def training_policy_2(cfg: TrainingConfig):
    training_policy_1(cfg, lr_init=5e-4)


def training_prepare(cfg: TrainingConfig, will_train=False, **kwargs):
    cfg.data = TrainingData(cfg)
    cfg.data.check_data_length()
    print_data_base_info(cfg)
    if 'plot_sample' in kwargs.keys() and kwargs['plot_sample']:
        plot_train_val_example(cfg)
        return
    cfg.model, cfg.model_cfg = prepare_model(cfg)  # Model can only be prepared after TrainingConfig is fully setup
    if 'save_model' in kwargs.keys() and kwargs['save_model']:
        return
    if 'summary' in kwargs.keys() and kwargs['summary']:
        formatted_summary(cfg)
        return
    cfg.print_and_log(get_model_key_params(cfg))
    cfg.data.do_transforms()
    # Dataloaders
    train_dataset = MyDataset(pd_data=cfg.data.df3['train'], transforms=None)
    test_dataset = MyDataset(pd_data=cfg.data.df3['test'], transforms=None)
    val_dataset = MyDataset(pd_data=cfg.data.df3['validation'], transforms=None)
    cfg.train_dataset, cfg.test_dataset, cfg.val_dataset = train_dataset, test_dataset, val_dataset
    train_dataloader = MyDataLoader(train_dataset, cfg.model_cfg, batch_size=cfg.batch,
                                    num_batches_per_epoch=len(train_dataset) + 4, is_train=True)
    # ⚠️，batch_size * num_batches_per_epoch > 366⚠️
    test_dataloader = MyDataLoader(test_dataset if not cfg.use_val else val_dataset, cfg.model_cfg, batch_size=8,
                                   num_batches_per_epoch=48, is_train=False)  # 8*48==384>366,
    # batch = next(iter(train_dataloader))
    accelerator = Accelerator()
    cfg.accelerator = accelerator
    cfg.device = accelerator.device
    cfg.model.to(cfg.device)
    if cfg.weights:
        cfg.model = load_weights_file(cfg, cfg.weights)
    cfg.model.to(cfg.device)
    dict_policy = {1: training_policy_1, 2: training_policy_2}
    training_policy = dict_policy.get(cfg.policy, 1)
    training_policy(cfg)  # Modify some params in cfg
    if will_train:
        cfg.model, existing_records = process_resume(cfg)
        prepare_log(cfg, existing_records)
        cfg.last_epoch = 0 if cfg.last_epoch is None else cfg.last_epoch
    cfg.model, cfg.optimizer, cfg.train_dataloader, cfg.test_dataloader = accelerator.prepare(cfg.model, cfg.optimizer,
                                                                                              train_dataloader,
                                                                                              test_dataloader)
    return cfg


def training_worker(cfg: TrainingConfig):
    model, optimizer, = cfg.model, cfg.optimizer
    train_dataloader, test_dataloader = cfg.train_dataloader, cfg.test_dataloader
    accelerator, device, config = cfg.accelerator, cfg.device, cfg.model_cfg
    for epoch in range(cfg.last_epoch + 1, cfg.epochs + 1):
        empty_gpu_cache(cfg)
        mase, smape = None, None
        cfg.current_epoch = epoch
        trange_train = tqdm.tqdm(train_dataloader, bar_format='{l_bar}{bar:30}{r_bar}')
        str_date = datetime.now().strftime("%m-%d %H:%M")
        lr = get_current_lr(cfg)
        running_loss = 0
        model.train()
        for batch_train in trange_train:
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch_train["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch_train["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch_train["past_time_features"].to(device),
                past_values=batch_train["past_values"].to(device),
                future_time_features=batch_train["future_time_features"].to(device),
                future_values=batch_train["future_values"].to(device),
                past_observed_mask=batch_train["past_observed_mask"].to(device),
                future_observed_mask=batch_train["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            accelerator.backward(loss)  # Backpropagation
            optimizer.step()
            running_loss += loss.cpu()
            average_loss = running_loss / (trange_train.n + 1)
            eft = f"\033[35m{tqdm_estimated_finish_time(trange_train)}\033[0m"
            bar_train = (f"\033[1;36m{str_date} {gpu_mem_in_use(cfg)} "
                         f"Epoch {epoch:3d}/{cfg.epochs:3d} lr={lr:.2e} loss={average_loss:.4f} {eft}")
            trange_train.set_description(bar_train)
            trange_train.refresh()  # to show immediately the update
            time.sleep(0.001)
        trange_train.close()
        del outputs, trange_train,
        empty_gpu_cache(cfg)
        if epoch >= cfg.save_from_epoch or cfg.debug:
            model.eval()
            trange_val = tqdm.tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}')
            forecasts_list = []
            for batch_val in trange_val:
                outputs = model.generate(
                    static_categorical_features=batch_val["static_categorical_features"].to(device)
                    if config.num_static_categorical_features > 0
                    else None,
                    static_real_features=batch_val["static_real_features"].to(device)
                    if config.num_static_real_features > 0
                    else None,
                    past_time_features=batch_val["past_time_features"].to(device),
                    past_values=batch_val["past_values"].to(device),
                    future_time_features=batch_val["future_time_features"].to(device),
                    past_observed_mask=batch_val["past_observed_mask"].to(device),
                )
                output_seq = outputs.sequences.cpu().numpy()
                forecasts_list.append(output_seq)
                eft = f"\033[35m{tqdm_estimated_finish_time(trange_val)}\033[0m"
                bar = f"\033[1;32m{str_date} {gpu_mem_in_use(cfg)} Epoch {epoch:3d}/{cfg.epochs:3d} {eft}"
                trange_val.set_description(bar)
                trange_val.refresh()  # to show immediately the update
                time.sleep(0.001)
            trange_val.close()
            forecasts = np.vstack(forecasts_list[:len(cfg.test_dataset)])
            del outputs, trange_val, forecasts_list
            empty_gpu_cache(cfg)
            mase, smape = calc_mase_and_smape(cfg, forecasts, cfg.test_dataset)
            with open(os.path.join(cfg.working_dir, cfg.results_dir, f"forecast-{epoch:03d}.dat"), "wb") as f:
                pickle.dump(forecasts, f)
        if epoch >= cfg.save_from_epoch or cfg.debug:  # 没有必要每个epoch保存结果，因为前期的epoch效果很差
            if mase and smape:
                file_name = f"epoch-{epoch:03d}-{average_loss:.4f}-{mase:.4f}-{smape:.4f}.pth"
            else:
                file_name = f"epoch-{epoch:03d}-{average_loss:.4f}.pth"
            weight_file = os.path.join(cfg.working_dir, cfg.results_dir, file_name)
            torch.save(model.state_dict(), weight_file)
        actions_after_epoch(cfg, bar_train, mase, smape)
        cfg.lr_scheduler.step()


def check_dataloader_worker(cfg: TrainingConfig, train=False, test=False):
    train_dataloader, test_dataloader = cfg.train_dataloader, cfg.test_dataloader
    loaders = []
    loaders.append(train_dataloader) if train else ()
    loaders.append(test_dataloader) if test else ()
    for loader in loaders:
        for epoch in range(1, 3):
            count = 0
            for batch in loader:
                count += 1
                print(f"Epoch {epoch}: batch-{count:2d} {batch['static_categorical_features'].view(-1)}")
            sep_line()
    print(f"{list(batch.keys())}")


def plot_forecast_worker(cfg: TrainingConfig, epoch: int, ts_index: int):
    with open(os.path.join(cfg.working_dir, cfg.results_dir, f"forecast-{epoch:03d}.dat"), "rb") as f:
        forecasts = pickle.load(f)  # np.ndarray
    mase, smape = calc_mase_and_smape(cfg, forecasts, cfg.test_dataset)
    title = f"Epoch {epoch} {mase:.4f} {smape:.4f}"
    plot_forecast(cfg, forecasts, ts_index, cfg.test_dataset, title)


# --validate results/weight-0100.pth 0
def val_worker(cfg: TrainingConfig, weights_file: str, ts_index: int):
    val_dataloader = MyDataLoader(cfg.val_dataset, cfg.model_cfg, batch_size=32,
                                  num_batches_per_epoch=12, is_train=False)  # 32*12==384, >366, 已经涵盖了所有数据
    batch = val_dataloader[ts_index]
    model, device = cfg.model, cfg.device
    model = load_weights_file(cfg, weights_file, adaptive=True)  # adaptive=True 将根据weights_file调整model参数
    config = cfg.model_cfg
    model.eval()
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    forecasts = outputs.sequences.cpu().numpy()  # forecasts = np.expand_dims(forecasts, axis=0)
    plot_forecast(cfg, forecasts, ts_index, cfg.val_dataset, title=f"Validation Series {ts_index} @ {weights_file}")


# endregion


# region ✅ 6 Program Entry
def parse_arguments():
    defaults = {'batch': 256, 'policy': 1, 'top_n_files': 10,
                'encoder_layers': 16, 'decoder_layers': 32, 'attention_heads': 8,
                'save_from_epoch': 10, 'mase_limit': 2.0}
    helps = {'sum': "\033[35mNo training, \033[0msummarize model and exit",
             'save_model': "\033[35mNo training, \033[0msave model and model configuration",
             'load': "Load existing weights file before training",
             'batch': "The batch size for training",
             'policy': "Training policy selection",
             'plot_sample': "\033[35mNo training, \033[0mplot sample data and exit",
             'plot': "\033[35mNo training, \033[0mplot Forecasts and exit, epoch and index must be provided",
             'save_from_epoch': "Start saving forecast result from epoch",
             'check_data': "\033[35mNo training, \033[0mcheck batch data, 1=train data, 2= test data, 3=both",
             'validate': ("\033[35mNo training, \033[0muse validation dataset to do some testing, "
                          "weights \33[35m(FULLPATH)\33[0m and ts_index MUST be provided"),
             'results_dir': "Assign results dir other than default, \033[35mrelative path to current\033[0m",
             'save_dir': "Assign save dir other than default, \033[35mrelative path to current\033[0m",
             'resume': ("\033[1;31mResume training, \033[0mlast lr in training record will be used or use --lr; "
                        "last weight in --results_dir will be used or use --weight \33[35m(FULLPATH)\33[0m"),
             'lr': "Learning rate to assign for training",
             'weights': "Initial weights file to use for training, \033[35mFULLPATH\033[0m",
             'top_n_files': "Best number of weights files to keep",
             'encoder_layers': "\033[1;31mNumber of encoder layers\033[0m",
             'decoder_layers': "\033[1;31mNumber of decoder layers\033[0m",
             'attention_heads': "\033[1;31mNumber of encoder and decoder attention_heads\033[0m",
             'early_stop': ("\033[1;31mDuring training, monitor \033[35mhow many epochs\033[0m "
                            "loss not decreased from best. "
                            "If specified, training process will stop when this condition is met"),
             'mase_limit': "Mase value higher than this will be thought as too high, and pth file will be dropped",
             }
    parser = argparse.ArgumentParser(description='American Presidential Vote Analysis', prog='apva',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 3.99 Build 2023-08-21')
    parser.add_argument("-D", "--debug", help="Print extra debug information", action="store_true")
    parser.add_argument("--encoder_layers", metavar="N", help=helps['encoder_layers'],
                        default=defaults['encoder_layers'], type=int)
    parser.add_argument("--decoder_layers", metavar="N", help=helps['decoder_layers'],
                        default=defaults['decoder_layers'], type=int)
    parser.add_argument("--attention_heads", metavar="N", help=helps['attention_heads'],
                        default=defaults['attention_heads'], type=int)
    parser.add_argument("-r", "--resume", help=helps['resume'], action="store_true")
    parser.add_argument("--early_stop", metavar='N', help=helps['early_stop'], type=int)
    parser.add_argument("--lr", help=helps['lr'], type=float)
    parser.add_argument("--weights", metavar="FILE", help=helps['weights'], type=str)
    # parser.add_argument("--test_step", metavar="N", help="Test every N steps", type=int, default=1)
    parser.add_argument("--mase_limit", metavar="FLOAT", help=helps['mase_limit'], type=float,
                        default=defaults['mase_limit'])
    parser.add_argument("--use_val", help="Use validation dataset for testing", action="store_true")
    parser.add_argument("-S", "--save_from_epoch", metavar='EPOCH', help=helps['save_from_epoch'],
                        default=defaults['save_from_epoch'], type=int)
    # Other args from this line, lines before should be kept
    parser.add_argument("--batch", metavar="N", help=helps['batch'], default=defaults['batch'], type=int)
    parser.add_argument("--policy", metavar="N", help=helps['policy'], default=defaults['policy'], type=int)
    parser.add_argument("--results_dir", metavar="DIR", help=helps['results_dir'], default='results', type=str)
    parser.add_argument("--save_dir", metavar="DIR", help=helps['save_dir'], default='save', type=str)
    parser.add_argument("--top_n_files", metavar="N", help=helps['top_n_files'],
                        default=defaults['top_n_files'], type=int)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_model", help=helps['save_model'], action="store_true")
    group.add_argument("--sum", help=helps["sum"], action="store_true")
    group.add_argument("--plot_sample", help=helps['plot_sample'], action="store_true")
    group.add_argument("--plot", nargs=2, metavar=('EPOCH', 'INDEX'), help=helps['plot'], type=int)
    group.add_argument("--validate", nargs=2, metavar=('WEIGHTS_FILE', 'TS_INDEX'), help=helps['validate'])
    group.add_argument("--check_data", choices=[1, 2, 3], help=helps['check_data'], type=int)
    return parser.parse_args()


def main(cfg: TrainingConfig):
    argv = parse_arguments()
    cfg.update_from(argv)
    cfg.update_working_path()
    [print(f'\033[32m{k:<20} -> {v}\033[0m') for k, v in vars(argv).items() if v]
    cfg.update_from(argv)
    early_exits_in_training_prepare = any([argv.save_model, argv.sum, argv.plot_sample])
    if early_exits_in_training_prepare:  # Those args will cause early exit
        training_prepare(cfg, save_model=True if argv.save_model else False,
                         summary=True if argv.sum else False,
                         plot_sample=True if argv.plot_sample else False,
                         will_train=False)
        return
    if argv.check_data:
        cfg = training_prepare(cfg)
        check_dataloader_worker(cfg, train=True if argv.check_data in [1, 3] else False,
                                test=True if argv.check_data in [2, 3] else False)
    elif argv.plot:
        cfg = training_prepare(cfg)
        epoch, index = argv.plot[0], argv.plot[1]
        plot_forecast_worker(cfg, epoch=epoch, ts_index=index)
    elif argv.validate is not None:
        cfg = training_prepare(cfg)
        val_worker(cfg, weights_file=argv.validate[0], ts_index=int(argv.validate[1]))
    else:
        print("\033[1;35mModel will start training in 30 seconds\033[0m")
        print(f"\033[1;31mWarning, all files in \033[0m{cfg.results_dir}\033[1;31m will be deleted\033[0m")
        for _ in tqdm.trange(30):  # Sleep 30 seconds just in case of disastrous params
            time.sleep(1)
        updated_cfg = training_prepare(cfg, will_train=True)
        training_worker(updated_cfg)


if __name__ == "__main__":
    configurations = TrainingConfig()
    try:
        main(configurations)
    except KeyboardInterrupt:
        early_exit_by_ctrl_c(configurations)
# endregion
