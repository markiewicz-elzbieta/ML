import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
import kagglehub
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import config
import warnings
import logging
import os


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

logging.getLogger("absl").setLevel("ERROR")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

np.random.seed(42)
tf.random.set_seed(42)


class PipelineStateError(Exception):
    """Raised when a pipeline method is called before required steps are completed."""


class LSTMPipeline:
    """
    End-to-end stateful pipeline for time series forecasting with LSTM.

    Each instance holds its own scalers, model, and training history,
    making it safe to run multiple configurations in parallel:

        pipeline_a = LSTMPipeline(neurons=[64, 32], window=10)
        pipeline_b = LSTMPipeline(neurons=[128, 64], window=20)
    """

    def __init__(
        self,
        neurons: list[int]   = config.NUMBER_OF_NEURONS,
        window: int          = config.SEQUENCE_WINDOW,
        dropout: float       = config.DROPOUT_FRACTION,
        learning_rate: float = config.LEARNING_RATE,
        loss: str            = config.LOSS,
        metrics: list[str]   = config.METRICS,
        batch_size: int      = config.BATCH_SIZE,
        epochs: int          = config.EPOCHS,
    ) -> None:
        """
        Initialize a new pipeline instance.

        Parameters
        ----------
        neurons : list[int], optional
            Number of neurons in each layer of the model, by default config.NUMBER_OF_NEURONS
        window : int, optional
            Sequence window size, by default config.SEQUENCE_WINDOW
        dropout : float, optional
            Dropout fraction, by default config.DROPOUT_FRACTION
        learning_rate : float, optional
            Learning rate, by default config.LEARNING_RATE
        loss : str, optional
            Loss function, by default config.LOSS
        metrics : list[str], optional
            List of metrics to track during training, by default config.METRICS
        batch_size : int, optional
            Batch size, by default config.BATCH_SIZE
        epochs : int, optional
            Number of training epochs, by default config.EPOCHS
        Returns
        -------
        None
        """
        self.neurons                = neurons
        self.window                 = window
        self.dropout                = dropout
        self.learning_rate          = learning_rate
        self.loss                   = loss
        self.metrics                = metrics
        self.batch_size             = batch_size
        self.epochs                 = epochs

        self.scaler_X: MinMaxScaler | None              = None
        self.scaler_y: MinMaxScaler | None              = None
        self.model: tf.keras.Model | None               = None
        self.history: tf.keras.callbacks.History | None = None
        

        logger.info(
            f"Pipeline created | neurons: {neurons} | window: {window} | dropout: {dropout}"
        )

    @staticmethod
    def download_dataset(kagglehub_path: str) -> pd.DataFrame:
        """
        Download a CSV dataset from Kaggle Hub and return it as a DataFrame.

        :param kagglehub_path: Kaggle dataset identifier (e.g. 'user/dataset-name')
        :raises OSError: If download fails or no CSV file is found
        """
        try:
            path = kagglehub.dataset_download(kagglehub_path)
        except Exception as e:
            raise OSError(f"Failed to download '{kagglehub_path}': {e}") from e

        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_path = os.path.join(path, file)
                logger.info(f"Dataset loaded")
                return pd.read_csv(csv_path)

        raise OSError(f"No CSV file found in downloaded dataset: {path}")
    
    @staticmethod
    def moving_average(df: pd.DataFrame, target_column: str, ma_window: int) -> pd.DataFrame:
        """
        Add a moving average column of target_column to the DataFrame.
        Window is capped at the number of available data points.
        Each value is computed from preceding values only (no lookahead).

        :param df: Prepared DataFrame, sorted chronologically
        :param target_column: Column to compute moving average from
        :param window: Number of preceding values to average over
        :return: DataFrame with additional column '{target_column}_ma_{window}'
        """
        return (
            df[target_column]
            .shift(1)
            .fillna(df[target_column].iloc[0])
            .rolling(window=ma_window, min_periods=1)
            .mean()
            )


    @staticmethod
    def prepare_dataset(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Select columns, parse dates, and sort chronologically.

        :param df: Raw DataFrame
        :param columns: Columns to keep; must include 'Date'
        :raises ValueError: If 'Date' is missing from columns or DataFrame
        """
        if "Date" not in columns:
            raise ValueError(f"'Date' must be in columns, got: {columns}")

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        return (
            df[columns]
            .copy()
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
            .sort_values("Date")
            .reset_index(drop=True)
        )

    @staticmethod
    def extract_features(
        df: pd.DataFrame,
        target_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split DataFrame into feature matrix X and target vector y.
        'Date' and target_column are excluded from X.

        :param df: Prepared DataFrame
        :param target_column: Column to predict
        :raises ValueError: If target_column not found in DataFrame
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column, "Date"], errors="ignore").to_numpy()
        y = df[target_column].to_numpy()
        X_columns = df.drop(columns=[target_column, "Date"], errors="ignore").columns.tolist()
        logger.info(
            f"Used features: {X_columns}"
        )

        return X, y, X_columns

    @staticmethod
    def split_dataset(
        data: np.ndarray,
        split_train: int,
        split_val: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chronological train / validation / test split.

        :param data: Full array (X or y)
        :param split_train: Index where validation begins
        :param split_val: Index where test begins
        """
        return data[:split_train], data[split_train:split_val], data[split_val:]

    def fit_scalers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers on training data and store them in pipeline state.
        Must be called before transform().

        :param X_train: shape (n, feat_num)
        :param y_train: shape (n,)
        :return: (X_train_scaled, y_train_scaled)
        """
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        return X_scaled, y_scaled

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform data using already-fitted scalers.
        Values outside the training range may exceed [0, 1] — this is expected.

        :param X: shape (n, feat_num)
        :param y: shape (n,)
        :raises PipelineStateError: If fit_scalers() has not been called yet
        """
        if self.scaler_X is None or self.scaler_y is None:
            raise PipelineStateError("Call fit_scalers() before transform().")

        return (
            self.scaler_X.transform(X),
            self.scaler_y.transform(y.reshape(-1, 1)),
        )

    def convert_to_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences for LSTM input.
        X[i] covers timesteps [i, i+window), y[i] is the target at i+window.

        :param X: Scaled features, shape (n, feat_num)
        :param y: Scaled target, shape (n, 1)
        :return: X_seq (n-window, window, feat_num), y_seq (n-window, 1)
        """
        X_seq = np.array([X[i:i + self.window] for i in range(len(X) - self.window)])
        y_seq = np.array([y[i + self.window]   for i in range(len(y) - self.window)])
        return X_seq, y_seq

    def build(self, feat_num: int) -> None:
        if not self.neurons:
            raise ValueError("neurons must contain at least one value.")

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((self.window, feat_num)))

        for i, units in enumerate(self.neurons):
            return_sequences = i < len(self.neurons) - 1
            model.add(tf.keras.layers.LSTM(units, return_sequences=return_sequences))
            if self.dropout > 0:
                model.add(tf.keras.layers.Dropout(self.dropout))

        model.add(tf.keras.layers.Dense(1))

        model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=self.metrics,
        )
        self.model = model
        logger.info(f"Model built | layers: {len(self.neurons)} | neurons: {self.neurons}")
        self.model.summary()


    def fit(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ) -> None:
        """
        Train the model and store history in pipeline state.

        :param callbacks: Optional list of Keras callbacks, e.g. EarlyStopping
        :raises PipelineStateError: If build() has not been called yet
        """
        if self.model is None:
            raise PipelineStateError("Call build() before fit().")

        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
        )

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Run inference and return inverse-transformed predictions.

        :param X_seq: Sequenced features, shape (n, window, feat_num)
        :raises PipelineStateError: If model or scalers are not ready
        :return: Predictions in original scale, shape (n, 1)
        """
        if self.model is None:
            raise PipelineStateError("Call build() and fit() before predict().")
        if self.scaler_y is None:
            raise PipelineStateError("Call fit_scalers() before predict().")

        return self.scaler_y.inverse_transform(self.model.predict(X_seq))
    

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> None:
        """
        Compute and log MAE, RMSE and MAPE for a single dataset split.

        :param y_true: Ground truth values, shape (n, 1)
        :param y_pred: Model predictions, shape (n, 1)
        :param split_name: Label for the split, e.g. 'Train', 'Val', 'Test'
        """
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        logger.info(
            f"{split_name} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%"
        )


    def plot_loss(self) -> None:
        """
        Plot train vs. validation curves for all configured metrics.

        :raises PipelineStateError: If fit() has not been called yet
        """
        if self.history is None:
            raise PipelineStateError("Call fit() before plot_loss().")

        for metric in self.metrics:
            plt.figure()
            plt.plot(self.history.history[metric],          label="train")
            plt.plot(self.history.history[f"val_{metric}"], label="validation")
            plt.title(
                f"Neurons: {self.neurons} | "
                f"Dropout: {self.dropout} | "
                f"Metric: {metric}"
            )
            plt.ylabel(metric)
            plt.xlabel("epoch")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_predictions(
        df: pd.DataFrame,
        y_train_pred: np.ndarray,
        y_val_pred: np.ndarray,
        y_test_pred: np.ndarray,
        split_train: int,
        split_val: int,
    ) -> None:
        """
        Interactive Plotly chart: OHLCV + predictions + split markers.

        :param df: Full DataFrame
        :param y_train_pred: Inverse-transformed train predictions, shape (n, 1)
        :param y_val_pred:   Inverse-transformed val predictions,   shape (n, 1)
        :param y_test_pred:  Inverse-transformed test predictions,  shape (n, 1)
        :param window: Sequence window (offsets prediction dates)
        :param split_train: Row index where validation begins
        :param split_val:   Row index where test begins
        """
        fig = go.Figure()

        for col, dash in [("Open", "solid"), ("Close", "solid"), ("Low", "dash"), ("High", "dash")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col, line=dict(dash=dash), yaxis="y1"))

        if "Volume" in df.columns:
            fig.add_trace(
                go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="darkgray", opacity=0.7, yaxis="y2")
            )

        y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
        dates_all  = df["Date"].iloc[len(df) - len(y_all_pred):].values

        fig.add_trace(
            go.Scatter(x=dates_all, y=y_all_pred.flatten(), name="Prediction",
                    line=dict(color="blue", dash="dot", width=1.5), yaxis="y1")
        )

        for idx, label in [(split_train, "Train/Val"), (split_val, "Val/Test")]:
            fig.add_vline(
                x=df["Date"].iloc[idx].value // 10 ** 6,
                line=dict(color="red", dash="dash", width=1.5),
                annotation_text=label,
                annotation_position="top",
            )

        fig.update_layout(
            title="Google Stock Prices Over Time",
            yaxis=dict(title="Price (USD)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            xaxis=dict(
                rangeselector=dict(buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
                rangeslider=dict(visible=True),
                type="date",
            ),
            hovermode="x unified",
        )
        fig.show()

    def plot_shap(
    self,
    X_train_seq: np.ndarray,
    X_explain_seq: np.ndarray,
    feature_names: list[str],
    n_background: int = 100,
    n_explain: int = 50,
    ) -> None:
        """
        Compute and plot SHAP values using GradientExplainer.
        SHAP values are averaged across the time (window) dimension
        to produce one importance score per feature.

        :param X_train_seq: Training sequences used as background dataset, shape (n, window, feat_num)
        :param X_explain_seq: Sequences to explain, shape (n, window, feat_num)
        :param feature_names: List of feature names matching feat_num
        :param n_background: Number of background samples for GradientExplainer (default 100)
        :param n_explain: Number of samples to explain (default 50)
        :raises PipelineStateError: If model has not been built and trained yet
        """

        if self.model is None:
            raise PipelineStateError("Call build() and fit() before plot_shap().")

        background  = X_train_seq[:n_background]
        explain     = X_explain_seq[:n_explain]

        weights     = self.model.get_weights()
        model_copy  = tf.keras.models.clone_model(self.model)
        model_copy.set_weights(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer   = shap.GradientExplainer(model_copy, background)
            shap_values = explainer.shap_values(explain)

        if isinstance(shap_values, list):
            shap_raw = shap_values[0]
        else:
            shap_raw = shap_values

        shap_raw = shap_raw.squeeze(-1)
        shap_values_mean = np.mean(np.abs(shap_raw), axis=1)

        shap.summary_plot(
            shap_values_mean,
            features=explain[:, -1, :],
            feature_names=feature_names,
            plot_type="dot",
        )



def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train and evaluate an LSTM pipeline on Google stock price data.

    Steps:
        1. Download and prepare the dataset from Kaggle Hub.
        2. Optionally add moving average features to the DataFrame.
        3. Extract features and split chronologically into train / val / test.
        4. Fit scalers on training data and transform all splits.
        5. Convert arrays into sliding-window sequences for LSTM input.
        6. Build, compile, and train the model.
        7. Plot train vs. validation loss curves.
        8. Generate predictions for all splits and visualize results.

    :return: Tuple of inverse-transformed predictions (y_train_pred, y_val_pred, y_test_pred),
             each of shape (n, 1), in the original target scale.
    """
    df = LSTMPipeline.download_dataset(config.KAGGLEHUB_PATH)
    df = LSTMPipeline.prepare_dataset(df, config.COLUMNS_LIST)
    if config.ADDITIONAL_MA_FEATURES:
        for ma in config.ADDITIONAL_MA_FEATURES:
            df[f"MA{str(ma)}"] = LSTMPipeline.moving_average(df, config.TARGET_COLUMN, ma)

    X, y, X_columns = LSTMPipeline.extract_features(df, config.TARGET_COLUMN)
    feat_num = X.shape[1]

    n = len(X)
    split_train = round(config.TRAIN_FRACTION * n)
    split_val = round(config.TRAIN_VAL_FRACTION * n)

    X_train, X_val, X_test = LSTMPipeline.split_dataset(X, split_train, split_val)
    y_train, y_val, y_test = LSTMPipeline.split_dataset(y, split_train, split_val)

    logger.info(
        f"Split | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} rows"
    )

    pipeline = LSTMPipeline(
        neurons=config.NUMBER_OF_NEURONS,
        window=config.SEQUENCE_WINDOW,
        dropout=config.DROPOUT_FRACTION,
        learning_rate=config.LEARNING_RATE,
        loss=config.LOSS,
        metrics=config.METRICS,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
    )

    X_train_sc, y_train_sc = pipeline.fit_scalers(X_train, y_train)
    X_val_sc, y_val_sc = pipeline.transform(X_val,  y_val)
    X_test_sc, y_test_sc = pipeline.transform(X_test, y_test)

    X_train_seq, y_train_seq = pipeline.convert_to_sequences(X_train_sc, y_train_sc)
    X_val_seq, y_val_seq = pipeline.convert_to_sequences(X_val_sc, y_val_sc)
    X_test_seq, _ = pipeline.convert_to_sequences(X_test_sc, y_test_sc)

    pipeline.build(feat_num)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    pipeline.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, callbacks=[early_stopping])
    pipeline.plot_loss()

    y_train_pred = pipeline.predict(X_train_seq)
    y_val_pred = pipeline.predict(X_val_seq)
    y_test_pred = pipeline.predict(X_test_seq)

    LSTMPipeline.plot_predictions(
        df, y_train_pred, y_val_pred, y_test_pred,
        split_train, split_val,
    )

    y_train_true = pipeline.scaler_y.inverse_transform(y_train_sc[pipeline.window:])
    y_val_true   = pipeline.scaler_y.inverse_transform(y_val_sc[pipeline.window:])
    y_test_true  = pipeline.scaler_y.inverse_transform(y_test_sc[pipeline.window:])

    LSTMPipeline.evaluate(y_train_true, y_train_pred, "Train")
    LSTMPipeline.evaluate(y_val_true,   y_val_pred,   "Val")
    LSTMPipeline.evaluate(y_test_true,  y_test_pred,  "Test")


    pipeline.plot_shap(
        X_train_seq=X_train_seq,
        X_explain_seq=X_test_seq,
        feature_names=X_columns,
        n_background=config.N_BACKGROUND,
        n_explain=config.N_EXPLAIN
    )

    return y_train_pred, y_val_pred, y_test_pred, pipeline.history


if __name__ == "__main__":
    main()