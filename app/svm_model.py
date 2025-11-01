import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from app.scaling import Preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM_Prediction(Preprocessing):

    def __init__(self, exchange, interval, asset, market = None):
        super().__init__(exchange, interval, asset, market)
        self.model = SVC(kernel = 'rbf', C = 1.0, random_state = 42)
        self.scaler = StandardScaler()

    def train_model(self):
        features = ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
                    'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'SEMA', 'LEMA', 'RSI', 'SR_K', 'SR_D',
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']

        df_action = self.df.copy()[features + ['Distinct_Action']].dropna()

        X = df_action[features].values
        y = df_action['Distinct_Action'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler.fit(X_train)

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.score_action = accuracy_score(y_test, y_pred) * 100

        self.save_model('models/svm_action_prediction_model.pkl')
        self.save_scaler('models/svm_scaler.pkl')

    def save_scaler(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, filepath):
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def get_prediction(self):
        self.load_model('models/svm_action_prediction_model.pkl')
        self.load_scaler('models/svm_scaler.pkl')
        features = ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
                    'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'SEMA', 'LEMA', 'RSI', 'SR_K', 'SR_D',
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']

        df_action = self.df.copy()[features].dropna()

        X_pred = self.scaler.transform(df_action.values)

        self.model_prediction_action = self.model.predict(X_pred)
        self.requested_prediction_action = self.model_prediction_action[-1]

    def prediction_postprocessing(self, indication):
        self.indication = indication
        indicators = {'Analysed':'Distinct_Action', 'Predicted':'Action_Predictions'}

        features = ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
            'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'SEMA', 'LEMA', 'RSI', 'SR_K', 'SR_D',
            'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']

        # Clean the DataFrame in the same way as in get_prediction to ensure data alignment
        self.df_visualization = self.df.dropna(subset=features).copy()
        self.df_visualization['Action_Predictions'] = self.model_prediction_action

        self.df_visualization = self.df_visualization[['Open', 'Adj Close', 'Volume', 'Distinct_Action', 'Action_Predictions']]

        self.df_visualization['Price_Buy'] = self.df_visualization[self.df_visualization[indicators[self.indication]] == 'Buy']['Adj Close']
        self.df_visualization['Price_Sell'] = self.df_visualization[self.df_visualization[indicators[self.indication]] == 'Sell']['Adj Close']

        self.df_visualization['Bullish Volume'] = self.df_visualization[self.df_visualization['Adj Close'] >= self.df_visualization['Open']]['Volume']
        self.df_visualization['Bearish Volume'] = self.df_visualization[self.df_visualization['Adj Close'] < self.df_visualization['Open']]['Volume']

        self.df_visualization_technical = self.df.dropna(subset=features)[['OBV', 'MACD', 'MACDS', 'MACDH', 'RSI', 'SR_K', 'SR_D', 'SR_RSI_K', 'SR_RSI_D', 'ATR']]
