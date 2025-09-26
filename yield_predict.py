# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 16:54
# @Author  : chengxiang.luo
# @Email   : chengxiang.luo@foxmail.com
# @File    : yield_predict.py
# @Software: PyCharm
import numpy as np
import torch
from matplotlib import pyplot as plt
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
)
from sklearn.model_selection import train_test_split
from rxn_logger import logger
from torch import nn, optim

from rxn_data_processing_utils import RxnDataProcessing


class YieldPredict(nn.Module):
    def __init__(self, input_dim=263, hidden_dim=128, output_size=2):
        super(YieldPredict, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)
        return out


class MultiHeadYieldPredict(nn.Module):
    def __init__(self, input_dim=263, hidden_dim=128):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        output1 = self.head1(shared_features)
        output2 = self.head2(shared_features)
        return output1, output2


class ReactionYieldPredictor:
    def __init__(self, df, model):
        self.model = model
        self.fingerprint_data = {}
        self.df = df
        self.df = self.df.reset_index(drop=True)

    def get_fingerprints(self, rxnfp_generator: RXNBERTFingerprintGenerator):
        self.df['fps'] = self.df['reaction_SMILES'].apply(lambda x: rxnfp_generator.convert(x))

    def gen_train_test_data(self):
        X_list = []
        y_list = []

        for _, row in self.df.iterrows():
            _x = [row['eqv1'], row['eqv2'], row['eqv3'], row['eqv4'], row['eqv5'],
                  row['reaction_temperature'], row['time']]
            X = np.concatenate((row['fps'], _x))
            y = np.array([row['product1_yield'], row['product2_yield']])
            X_list.append(np.array(X))
            y_list.append(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_list, y_list, test_size=0.3, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, num_epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), "yield_predictor.pth")
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        return self.model

    @torch.no_grad()
    def evaluate(self, X_test, y_test):
        criterion = nn.MSELoss()
        outputs = model(X_test)
        _loss = criterion(outputs, y_test)
        logger.info(f'Test Loss: {_loss.item():.4f}')

        predicted_yields = outputs.numpy()
        logger.info(f'Predicted yields: {predicted_yields}')

    def plot_predictions(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([0, 100], [0, 100], 'r--', lw=2)
        plt.xlabel('True Yield (%)')
        plt.ylabel('Predicted Yield (%)')
        plt.title('Reaction Yield Prediction Performance')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def test_data_processing(file_path):
    rxn_process = RxnDataProcessing(file_path)
    df = rxn_process.csv_data_loader()
    rxn_process.plot_scatter()
    return df


def test_train_model(file_path, model, rxnfp_generator):
    df = test_data_processing(file_path)
    predictor = ReactionYieldPredictor(df, model)
    predictor.get_fingerprints(rxnfp_generator)
    X_train, X_test, y_train, y_test = predictor.gen_train_test_data()
    # train ane evaluate
    predictor.train(X_train, y_train, 20)
    predictor.evaluate(X_test, y_test)


def test_predict_yield(model, new_rxn_fp, extra_features):
    model.load_state_dict(torch.load('yield_predictor.pth'))
    model.eval()
    concat_feature = np.concatenate([new_rxn_fp, extra_features])
    tensor = torch.from_numpy(concat_feature).float()
    return model(tensor)


if __name__ == '__main__':
    model = YieldPredict()
    # model = MultiHeadYieldPredict()
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)

    test_data_processing('test_organic_reactions.csv')
    #
    test_train_model('test_organic_reactions.csv', model, rxnfp_generator)

    # predict
    rxn_smiles = ''
    # extra_features : 'eqv1', 'eqv2', 'eqv3', 'eqv4', 'eqv5', 'reaction_temperature', 'time',
    extra_features = [0.5916, 0.8326, 0.3226, 0.1662, 0.4036, 60.10, 27]
    new_rxn_fp = rxnfp_generator.convert(rxn_smiles)
    predicted_yield = test_predict_yield(model, new_rxn_fp, extra_features)
    print(predicted_yield)
