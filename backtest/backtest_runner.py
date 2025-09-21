# backtest/backtest_runner.py
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import numpy as np

# Assure qu'on peut importer depuis la racine du projet
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from strategies.strategy_rsi_macd_bbands import StrategyRSIMACDBBands


# =========================
#   CONFIG BACKTEST
# =========================
SYMBOL = os.getenv("BT_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BT_INTERVAL", "1h")
NB_CANDLES = int(os.getenv("BT_NB_CANDLES", "1500"))
INITIAL_CAPITAL = float(os.getenv("BT_INITIAL_CAPITAL", "1000"))
RISK_PER_TRADE = float(os.getenv("BT_RISK_PCT", "0.02"))  # 2% du capital par trade
SL_PCT = float(os.getenv("BT_SL_PCT", "0.02"))            # SL 2%
TP_PCT = float(os.getenv("BT_TP_PCT", "0.05"))            # TP 5%
SEED = int(os.getenv("BT_SEED", "42"))                    # pour reproductibilité
PRICE_DECIMALS = 2                                        # affichage


# =========================
#   LOGGING
# =========================
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


# =========================
#   UTILITAIRES
# =========================
def format_price(x: float) -> str:
    return f"{x:.{PRICE_DECIMALS}f}"


def load_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Source de données locale / fallback:
    - Si le projet a déjà un chargeur maison, remplace ici.
    - Pour rester offline et reproductible, on simule une série OHLC
      cohérente à partir d'un bruit contrôlé par SEED.
    Remarque: si tu veux vraiment tirer de Binance, remplace par ton fetcher.
    """
    rng = np.random.default_rng(SEED)
    # Crée un index temporel horaire récent
    end = pd.Timestamp.utcnow().floor("h")
    idx = pd.date_range(end=end, periods=limit, freq="H")

    # Génère un prix "drift + bruit" réaliste (base ~ 110k pour coller à tes logs)
    price = 110000 + np.cumsum(rng.normal(0, 200, size=limit))
    high = price + np.abs(rng.normal(80, 50, size=limit))
    low = price - np.abs(rng.normal(80, 50, size=limit))
    open_ = np.r_[price[0], price[:-1]]
    close = price

    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(50, 250, size=limit),
        }
    )
    df = df.reset_index(drop=True)
    return df


@dataclass
class Position:
    side: str           # "LONG" ou "SHORT" (on n'utilise que LONG ici)
    entry: float
    qty: float
    sl: float
    tp: float
    opened_at: pd.Timestamp


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float,
        risk_per_trade: float,
        sl_pct: float,
        tp_pct: float,
    ):
        self.df = df.copy()
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.pos: Optional[Position] = None
        self.stats: Dict[str, int | float] = {
            "wins": 0,
            "losses": 0,
            "net_profit": 0.0,
        }

    def _position_size(self, price: float) -> float:
        """Taille de position basée sur le risque fixe (montant au SL ~= risk_per_trade*capital)."""
        risk_usdt = max(1e-9, self.capital * self.risk_per_trade)
        sl_price = price * (1 - self.sl_pct)
        risk_per_unit = max(1e-9, price - sl_price)  # diff ABS
        qty = risk_usdt / risk_per_unit
        return qty

    def _open_long(self, ts: pd.Timestamp, price: float):
        qty = self._position_size(price)
        sl = price * (1 - self.sl_pct)
        tp = price * (1 + self.tp_pct)
        self.pos = Position("LONG", price, qty, sl, tp, ts)
        logger.info(
            f"[{ts:%Y-%m-%d %H:%M:%S}] OUVERTURE LONG @ {format_price(price)}, "
            f"SL=@{format_price(sl)}, TP=@{format_price(tp)} | qty={qty:.6f} BTC"
        )

    def _close_position(self, ts: pd.Timestamp, exit_price: float, reason: str):
        if not self.pos:
            return
        pnl = (exit_price - self.pos.entry) * self.pos.qty
        self.capital += pnl
        self.stats["net_profit"] = self.capital - self.initial_capital
        win = pnl > 0
        if win:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        logger.info(
            f"[{ts:%Y-%m-%d %H:%M:%S}] {reason} @ {format_price(exit_price)} | "
            f"qty={self.pos.qty:.6f} | PnL: {pnl:.2f} USDT"
        )
        self.pos = None

    def _check_sl_tp_intrabar(self, ts: pd.Timestamp, high: float, low: float):
        """Gestion intrabar (ordre: SL puis TP ou inverse ?)
        Hypothèse prudente: en LONG, si Low <= SL => SL touché avant TP.
        Si TP et SL sont tous deux franchis sur la même bougie, on déclenche SL en premier.
        """
        if not self.pos:
            return
        if self.pos.side == "LONG":
            # SL d'abord (hypothèse prudente)
            if low <= self.pos.sl:
                self._close_position(ts, self.pos.sl, "STOP LOSS")
                return
            if high >= self.pos.tp:
                self._close_position(ts, self.pos.tp, "TAKE PROFIT")
                return

    def run(self, strategy):
        logger.info("Démarrage du backtest avec SL/TP...")

        # Calcul des indicateurs requis par la stratégie
        self.df = strategy.prepare(self.df)

        # Boucle bougie par bougie (on commence après le warmup de la stratégie)
        for i in range(strategy.start_index, len(self.df)):
            row = self.df.iloc[i]
            ts = pd.to_datetime(row["open_time"])
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # Vérifie d'abord SL/TP intrabar sur position ouverte
            self._check_sl_tp_intrabar(ts, h, l)
            price_for_signal = c  # on prend la clôture pour le signal
            signal = strategy.signal(self.df, i)

            logger.info(f"[{ts:%Y-%m-%d %H:%M:%S}] Signal: {signal} | Prix: {format_price(price_for_signal)}")

            # Gestion des entrées/sorties déterministes:
            # - Si LONG ouvert: on ne sort que sur SL/TP, PAS sur signal inverse (cohérence des runs).
            # - Si pas de position: on n'entre que sur BUY.
            if self.pos is None and signal == "BUY":
                self._open_long(ts, price_for_signal)

        # Si encore en position à la fin, on clôture au dernier close (raison: CLOSE)
        if self.pos:
            last_ts = pd.to_datetime(self.df.iloc[-1]["open_time"])
            last_close = float(self.df.iloc[-1]["close"])
            self._close_position(last_ts, last_close, "CLOSE")

        # Résumé
        logger.info("===== Résultats du Backtest =====")
        logger.info(f"Capital initial : {self.initial_capital:.2f} USDT")
        logger.info(f"Capital final   : {self.capital:.2f} USDT")
        logger.info(f"Profit net      : {self.capital - self.initial_capital:.2f} USDT")
        logger.info(f"Trades gagnants : {self.stats['wins']}")
        logger.info(f"Trades perdants : {self.stats['losses']}")
        total_trades = self.stats["wins"] + self.stats["losses"]
        success = 100 * (self.stats["wins"] / total_trades) if total_trades > 0 else 0.0
        logger.info(f"Taux de réussite : {success:.2f}%")

        return self.stats


def main():
    # Message de connexion (pour coller aux logs que tu as montrés)
    logger.info("Connexion Binance établie en mode TESTNET")
    logger.info(f"Téléchargement de {NB_CANDLES} bougies en {INTERVAL} pour {SYMBOL}")
    logger.info(f"Récupération des données de {SYMBOL} en {INTERVAL}")

    # Charge les données (remplace par ton fetcher réel si nécessaire)
    df = load_binance_klines(SYMBOL, INTERVAL, NB_CANDLES)

    # Instancie la stratégie
    strategy = StrategyRSIMACDBBands()

    # Lance le backtest
    bt = Backtester(
        df=df,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
    )
    bt.run(strategy)


if __name__ == "__main__":
    main()
