import sys
import os

import pandas as pd
import streamlit as st


class LoadDatasetCheck:
    def __init__(self, filename, targetvalue_column=None):
        self.filename = filename
        self.targetvalue_column = targetvalue_column
        self.df = pd.DataFrame()
        self.classes = None
        self.target_values = None

        if self.filename is None:
            raise ValueError('input samples filename must be given!\n')

        if self.targetvalue_column is not None:
            if not (isinstance(targetvalue_column, str) or isinstance(targetvalue_column, int)):
                raise ValueError(f'the type of target value column={targetvalue_column} must be str or int\n')

    def load_dataset(self, set_index_col=None, dropna_row=True):
        file_extension = os.path.splitext(self.filename)[1]
        if file_extension == ".xlsx" or file_extension == ".xls":
            self.df = pd.read_excel(self.filename, index_col=set_index_col)
        elif file_extension == ".csv":
            self.df = pd.read_csv(self.filename, index_col=set_index_col)
        elif file_extension == ".txt":
            self.df = pd.read_csv(self.filename, sep='\t', index_col=set_index_col)
        else:
            raise NotImplementedError("The extension of \".xlsx\" and \".xls\" and \".csv\" and \".txt\" "
                                      "can only be supproted. User provided unknown format.")
        if self.df.empty:
            raise ValueError(f'{self.filename} is empty. please cheack it!')
        if dropna_row:
            self.df = self.df.dropna()
            if self.df.dropna().empty:
                raise ValueError(f'Too many missing values in {self.filename}. please cheack it!')
        return self


def load_dataset(filename, set_index_col=None, dropna_row=False):
    file_extension = os.path.splitext(filename.name)[1]
    if file_extension == ".xlsx" or file_extension == ".xls":
        df = pd.read_excel(filename, index_col=set_index_col)
    elif file_extension == ".csv":
        df = pd.read_csv(filename, index_col=set_index_col)
    elif file_extension == ".txt":
        df = pd.read_csv(filename, sep='\t', index_col=set_index_col)
    else:
        raise NotImplementedError("The extension of \".xlsx\" and \".xls\" and \".csv\" and \".txt\" "
                                  "can only be supproted. User provided unknown format.")
    if df.empty:
        raise ValueError(f'{filename} is empty. please cheack it!')
    if dropna_row:
        df = df.dropna()
        if df.dropna().empty:
            raise ValueError(f'Too many missing values in {filename}. please cheack it!')
    return df


