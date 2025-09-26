# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 13:54
# @Author  : chengxiang.luo
# @Email   : chengxiang.luo@foxmail.com
# @File    : rxn_data_processing_utils.py
# @Software: PyCharm

import pandas as pd
from matplotlib import pyplot as plt

from rxn_logger import logger


class RxnDataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = []
        self.yields_dict = {}
        self.temp_dict = {}

    def csv_data_loader(self, chunk_size=10000):
        """
        Read and process large CSV files in chunks, avoid memory overflow
        """
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            processed_chunk = self.chunk_standardization(chunk)
            self.create_dicts(processed_chunk)
            self.results.append(processed_chunk)
            logger.info(
                f'f"Finished processing chunk {len(self.results)}, containing {len(processed_chunk)} rows"')

        final_df = pd.concat(self.results, ignore_index=True)
        return final_df

    @staticmethod
    def chunk_standardization(chunk):
        chunk = chunk[chunk['reaction_SMILES'].notna() & (chunk['reaction_SMILES'] != '')]
        # remove unexpected yield rows
        numeric_fields = ['eqv1', 'eqv2', 'eqv3', 'eqv4', 'eqv5', 'reaction_temperature', 'time',
                          'product1_yield', 'product2_yield']

        # change fields to numeric, none as NaN
        for field in numeric_fields:
            chunk[field] = pd.to_numeric(chunk[field], downcast='float', errors='coerce')

        chunk['reaction_ID'] = pd.to_numeric(chunk['reaction_ID'], downcast='signed',
                                             errors='coerce')

        chunk = chunk[
            (chunk['product1_yield'] >= 0) & (chunk['product1_yield'] <= 100) &
            (chunk['product2_yield'] >= 0) & (chunk['product2_yield'] <= 100)
            ]

        chunk = chunk.dropna(subset=['reaction_ID'])
        # when 'eqv1', 'eqv2', 'eqv3', 'eqv4', 'eqv5', 'reaction_temperature'
        # is none, delete row from chunk
        chunk = chunk.dropna(
            subset=['eqv1', 'eqv2', 'eqv3', 'eqv4', 'eqv5', 'reaction_temperature'])

        return chunk

    def create_dicts(self, chunk):
        # create yields and temperature dicts
        for _, reaction in chunk.iterrows():
            reaction_id = reaction['reaction_ID']
            self.yields_dict[reaction_id] = [reaction['product1_yield'], reaction['product2_yield']]
            self.temp_dict[reaction_id] = reaction['reaction_temperature']

    def plot_scatter(self):
        # Prepare data: each reaction has two yields, so temperature and yield lists need to be expanded accordingly
        temperatures = []
        yields = []
        for rid, temperature in self.temp_dict.items():
            reaction_yields = self.yields_dict[rid]
            temperatures.extend([temperature] * len(reaction_yields))
            yields.extend(reaction_yields)

        plt.figure(figsize=(10, 6))
        plt.scatter(temperatures, yields, alpha=0.6, s=50)
        plt.xlabel('Reaction Temperature', fontsize=12)
        plt.ylabel('Product Yield (%)', fontsize=12)
        plt.title('Correlation between Reaction Temperature and Product Yield', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    rxn_process = RxnDataProcessing('test_organic_reactions.csv')
    df = rxn_process.csv_data_loader()
    print(df.head(10))
    rxn_process.plot_scatter()
