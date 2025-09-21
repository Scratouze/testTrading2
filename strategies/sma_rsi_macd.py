# strategies/strategy_rsi_macd_bbands.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import pandas as pd
import numpy as np


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal = ema(macd_line, signal_len)
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger(series: pd.Series, length: int = 20, num_std: float = 2.0):
    ma = series.rolling(length, min_periods=length).mean()
    std = series.rolling(length, min_periods=length).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


@dataclass
class StrategyRSIMACDBBands:
    """
    Stratégie simple, LONG only:
    - Filtre de tendance: MACD > Signal
    - Timing: RSI sort d'une zone neutre vers le haut ET clôture au-dessus de la bande moyenne Bollinger
    - On renvoie un signal discret: "BUY" / "SELL" / "HOLD"
      (Le runner n'entre que sur BUY et ne sort que par SL/TP => cohérence entre runs)
    """

    # paramètres
    rsi_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # index de départ après warm-up
    start_index: int = 100  # pour être safe avec tous les indicateurs

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].astype(float)

        df["rsi"] = rsi(close, self.rsi_len)
        df["macd"], df["macd_sig"], df["macd_hist"] = macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger(
            close, self.bb_len, self.bb_std
        )

        # drapeaux utiles
        df["macd_up"] = df["macd"] > df["macd_sig"]
        df["above_mid_bb"] = close > df["bb_mid"]

        return df

    def signal(self, df: pd.DataFrame, i: int) -> str:
        """
        BUY si:
          - tendance haussière (MACD > signal)
          - RSI > 55 (momentum)
          - clôture au-dessus de la moyenne BB
          - et un petit déclencheur: RSI(i-1) <= 55 & RSI(i) > 55 (cross up)
        SELL si inverse clair (non utilisé par le runner pour clôturer, mais utile pour debug)
        Sinon HOLD.
        """
        if i < 1:
            return "HOLD"

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Conditions BUY
        cond_trend = bool(row["macd_up"])
        cond_momentum = (prev["rsi"] <= 55) and (row["rsi"] > 55)
        cond_bb = bool(row["above_mid_bb"])

        if cond_trend and cond_momentum and cond_bb:
            return "BUY"

        # Conditions SELL (indicatif seulement)
        cond_trend_down = not bool(row["macd_up"])
        cond_momentum_down = (prev["rsi"] >= 45) and (row["rsi"] < 45)
        cond_bb_down = not bool(row["above_mid_bb"])

        if cond_trend_down and cond_momentum_down and cond_bb_down:
            return "SELL"

        return "HOLD"
