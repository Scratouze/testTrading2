# strategies/alpha_combo.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal

Signal = Literal["BUY", "SELL", "HOLD"]


# =========================
# Config de la stratégie
# =========================
@dataclass
class StrategyConfig:
    # Trend / momentum
    ema_fast: int = 21
    ema_slow: int = 55
    trend_sma: int = 200
    rsi_len: int = 14
    # Seuils assouplis pour générer plus d'entrées
    rsi_buy: float = 58.0
    rsi_sell: float = 42.0

    # Volatilité / niveaux
    atr_len: int = 14
    atr_mult_sl: float = 2.5
    atr_mult_tp: float = 3.2
    trail_atr_mult: float = 1.0

    # Renforcement du signal
    adx_len: int = 14
    adx_min: float = 23.0
    be_rr: float = 1.0          # break-even à 1R
    slope_lookback: int = 5

    # Sécurité
    min_bars: int = 300

    # Filtres de session / volatilité (BTC est 24/7 → on ouvre)
    trade_start_hour: int = 0
    trade_end_hour: int = 23
    avoid_weekends: bool = False
    atrp_window: int = 200
    atrp_min_quantile: float = 0.25


# =========================
# Utilitaires indicateurs
# =========================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr

def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _atr(df, 1)
    tr_smooth = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr_smooth.replace(0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx.fillna(0.0)


# =========================
# Stratégie
# =========================
class AlphaComboStrategy:
    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()

    def compute_levels(self, sig: Signal, close_px: float, atr: float) -> Dict[str, float]:
        c = self.cfg
        if sig == "BUY":
            sl = close_px - c.atr_mult_sl * atr
            tp = close_px + c.atr_mult_tp * atr
            trail_trigger = close_px + (c.atr_mult_tp * atr) * 0.25
        else:
            sl = close_px + c.atr_mult_sl * atr
            tp = close_px - c.atr_mult_tp * atr
            trail_trigger = close_px - (c.atr_mult_tp * atr) * 0.25
        return {"sl": float(sl), "tp": float(tp), "trail_trigger": float(trail_trigger)}

    def compute_be_trigger(self, sig: Signal, entry: float, sl: float) -> float:
        """Niveau de déclenchement break-even (1R par défaut)."""
        r = abs(entry - sl)
        if self.cfg.be_rr <= 0:
            return np.nan
        if sig == "BUY":
            return entry + self.cfg.be_rr * r
        else:
            return entry - self.cfg.be_rr * r

    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne un DataFrame indexé par datetime avec au minimum:
        ['open','high','low','close','ema_fast','ema_slow','sma_trend',
         'rsi','atr','adx','signal','tradable'(bool)]
        """
        df = candles.copy()
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            raise ValueError("Colonnes requises: open, high, low, close")

        c = self.cfg

        # Indicateurs
        df["ema_fast"] = _ema(df["close"], c.ema_fast)
        df["ema_slow"] = _ema(df["close"], c.ema_slow)
        df["sma_trend"] = _sma(df["close"], c.trend_sma)
        df["rsi"] = _rsi(df["close"], c.rsi_len)
        df["atr"] = _atr(df[["high", "low", "close"]], c.atr_len)
        df["adx"] = _adx(df[["high", "low", "close"]], c.adx_len)

        # Pente (EMA rapide)
        df["ema_slope"] = df["ema_fast"] - df["ema_fast"].shift(c.slope_lookback)

        # États de tendance
        df["trend_long"] = (df["close"] > df["sma_trend"]) & (df["ema_fast"] > df["ema_slow"]) & (df["ema_slope"] > 0)
        df["trend_short"] = (df["close"] < df["sma_trend"]) & (df["ema_fast"] < df["ema_slow"]) & (df["ema_slope"] < 0)
        df["strong"] = df["adx"] >= c.adx_min

        # Conditions brutes
        cond_buy = df["trend_long"] & df["strong"] & (df["rsi"] >= c.rsi_buy)
        cond_sell = df["trend_short"] & df["strong"] & (df["rsi"] <= c.rsi_sell)

        # --- Filtre horaire / week-end / volatilité ---
        hours = df.index.hour
        weekdays = df.index.weekday
        in_hours = (hours >= c.trade_start_hour) & (hours <= c.trade_end_hour)
        not_weekend = (weekdays <= 4) if c.avoid_weekends else True

        atr_low_thresh = df["atr"].rolling(c.atrp_window, min_periods=c.atrp_window).quantile(c.atrp_min_quantile)
        vol_ok = df["atr"] >= atr_low_thresh.fillna(np.inf)

        df["tradable"] = in_hours & not_weekend & vol_ok

        signal = np.where(cond_buy, "BUY", np.where(cond_sell, "SELL", "HOLD"))
        df["signal"] = pd.Series(signal, index=df.index).astype("string")

        if len(df) < c.min_bars:
            return df.iloc[0:0]
        return df.iloc[c.min_bars:].copy()
