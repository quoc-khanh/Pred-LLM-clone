import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

def format_feature_value(feature, value, row):
    # Automatically determine if the feature is numerical by checking its type
    # Assuming the row.columns[i] is a list of column values and checking the type of the first element
    try:
        first_value = row.columns[row.column_names.index(feature)].to_pylist()[0]
        # Check if the first value is numerical (either int or float)
        if isinstance(first_value, (int, float)) and value != 'None':
            s = "%s is" % feature
            if '.' in value:
                tmp = value.split('.')[1]
                tmp = min(3, len(tmp))  # Limit to 3 decimal places
                value = f"%.{tmp}f" % float(value)
            for v in value:
                s += ' %s' % v
            return s
        else:
            return "%s is %s" % (feature, value)
    except:
        return "%s is %s" % (feature, value)

class PredLLMDataset(Dataset):
    def get_ds_size(self, len):
        self.len = len

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        if key < self.len:
            # permute only feature variables
            all_column_idx = list(range(row.num_columns))
            shuffle_idx = all_column_idx[:-1]
            random.shuffle(shuffle_idx)
            # keep target variable at the end
            target_idx = all_column_idx[-1]
            # TODO
            # shuffled_text = ", ".join(
            #     ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
            # )
            # shuffled_text += ", {} is {}".format(row.column_names[target_idx],
            #                                      str(row.columns[target_idx].to_pylist()[0]).strip())
            shuffled_text = ", ".join(
                [format_feature_value(row.column_names[i], str(row.columns[i].to_pylist()[0]).strip(), row) for i in shuffle_idx]
            )
            shuffled_text += ", {} is {}".format(row.column_names[target_idx],
                                                str(row.columns[target_idx].to_pylist()[0]).strip())

        else:
            # fix both feature and target variables
            # shuffle_idx = list(range(row.num_columns))

            # shuffled_text = ", ".join(
            #     ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
            # )
            shuffle_idx = list(range(row.num_columns))
            
            # Create shuffled_text using format_feature_value for proper formatting
            shuffled_text = ", ".join(
                [format_feature_value(row.column_names[i], str(row.columns[i].to_pylist()[0]).strip(), row) for i in shuffle_idx]
            )

        # print("shuffled_text: {}".format(shuffled_text))
        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
