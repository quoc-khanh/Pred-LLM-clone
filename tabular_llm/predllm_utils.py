import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data

#TODO 
# def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
#     """ Converts the sentences back to tabular data

#     Args:
#         text: List of the tabular data in text form
#         df_gen: Pandas DataFrame where the tabular data is appended

#     Returns:
#         Pandas DataFrame with the tabular data from the text appended
#     """
#     columns = df_gen.columns.to_list()
        
#     # Convert text to tabular data
#     for t in text:
#         features = t.split(",")
#         td = dict.fromkeys(columns)
        
#         # Transform all features back to tabular data
#         for f in features:
#             values = f.strip().split(" is ")
#             if values[0] in columns and not td[values[0]]:
#                 try:
#                     td[values[0]] = [values[1]]
#                 except IndexError:
#                     #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
#                     pass
                
#         df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
#     return df_gen

def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data, removing spaces for numerical values only

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    columns = [c.strip() for c in columns]

    # Function to remove spaces for numerical values
    def clean_numerical_value(value: str) -> str:
        # Try to convert to float and if successful, remove spaces
        try:
            # Removing spaces only for numerical values
            return value.replace(" ", "")
        except ValueError:
            return value  # If it's not a numerical value, return as is

    # Convert text to tabular data
    df_list = [df_gen]
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)
        
        # Transform features back into tabular data
        for f in features:
            values = f.strip().split(" is ")
            values[0] = values[0].strip()  # Clean up the feature name
            
            # Check if the feature exists in columns and is not already populated
            if values[0] in columns and not td[values[0]]:
                if len(values) > 1:
                    value = values[1]  # Keep value as it is
                    
                    # Clean numerical values by removing spaces
                    value = clean_numerical_value(value)
                    
                    # Assign the value to the corresponding column
                    td[values[0]] = [value]
        
        # Append the transformed row to the DataFrame list
        df_list.append(pd.DataFrame(td))

    # Concatenate all rows back into a single DataFrame
    df_gen = pd.concat(df_list, ignore_index=True, axis=0)
    return df_gen



def _encode_row_partial(row, shuffle=True):
    """Function that takes a row and converts all columns into the text representation that are not NaN."""
    num_cols = len(row.index)
    if not shuffle:
        idx_list = np.arange(num_cols)
    else:
        idx_list = np.random.permutation(num_cols)

    lists = ", ".join(
        sum(
            [
                [f"{row.index[i]} is {row[row.index[i]]}"]
                if not pd.isna(row[row.index[i]])
                else []
                for i in idx_list
            ],
            [],
        )
    )
    return lists

