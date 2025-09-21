# backtest/backtest_runner.py
import logging
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Any
import pandas as pd
from pathlib import Path
import os

from strategies.alpha_combo import StrategyConfig, AlphaComboStrategy, Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# =========================
# Paramètres du backtest
# =========================
@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"

    # Trading
    risk_per_trade: float = 0.005   # 0.5% par trade (un peu plus agressif)
    max_leverage: float = 1.0
    fees_bps: float = 6.0
    slippage_bps: float = 2.0

    # Période
    start: Optional[str] = None
    end: Optional[str] = None

    # Source données
    data_csv: Optional[str] = None
    allow_short: bool = False

    # Anti-overtrading
    cooldown_bars: int = 4
    lockout_bars_after_loss: int = 6


# =========================
# Téléchargement via ccxt (fallback)
# =========================
def _download_with_ccxt(csv_path: Path, symbol_ccxt: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1500) -> pd.DataFrame:
    try:
        import ccxt, time
    except Exception:
        raise RuntimeError("ccxt n'est pas installé. Fais:  pip install ccxt")

    os.makedirs(csv_path.parent, exist_ok=True)
    exchange = ccxt.binance({"enableRateLimit": True})

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    # 3 ans en arrière
    since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365*3)).timestamp() * 1000)

    log.info(f"Téléchargement paginé de bougies {timeframe} pour {symbol_ccxt} depuis ~3 ans (ccxt)")
    all_rows = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_rows += ohlcv
        since = ohlcv[-1][0] + tf_ms
        # petite pause anti-rate-limit
        time.sleep(exchange.rateLimit / 1000)
        # garde une marge pour éviter la boucle infinie
        if len(all_rows) >= 26000:  # ≈ 3 ans de bougies 1h
            break

    if not all_rows:
        raise RuntimeError("Téléchargement vide depuis ccxt.")

    df = pd.DataFrame(all_rows, columns=["datetime_ms", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime_ms"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_csv(csv_path, index=True)  # garde l'index datetime en CSV
    log.info(f"CSV enregistré: {csv_path.as_posix()} ({len(df)} lignes)")
    return df


# =========================
# Chargement des données
# =========================
def load_candles(cfg: BacktestConfig) -> pd.DataFrame:
    if cfg.data_csv:
        path = Path(cfg.data_csv)
    else:
        path = Path("data") / f"{cfg.symbol}_{cfg.timeframe}.csv"

    if not path.exists():
        symbol_ccxt = "BTC/USDT" if cfg.symbol.upper() == "BTCUSDT" else cfg.symbol.replace("USDT", "/USDT")
        df = _download_with_ccxt(path, symbol_ccxt=symbol_ccxt, timeframe=cfg.timeframe, limit=1500)
    else:
        df = pd.read_csv(path)
        # détection colonne temps
        time_col = None
        for c in df.columns:
            lc = c.lower()
            if lc.startswith("time") or lc.startswith("date"):
                time_col = c
                break
        if time_col is None:
            time_col = "datetime"
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes (trouvées: {list(df.columns)}), requis: {required}")

    return df[["open", "high", "low", "close"]].copy()


# =========================
# Aide: coûts & exécution
# =========================
def apply_slippage(price: float, bps: float, side: Signal, direction: str) -> float:
    mult = 1.0 + (bps / 10000.0)
    inv_mult = 1.0 / mult
    if direction == "entry":
        return price * (mult if side == "BUY" else inv_mult)
    else:
        return price * (inv_mult if side == "BUY" else mult)

def fees_value(notional: float, bps: float) -> float:
    return abs(notional) * (bps / 10000.0)


# =========================
# Structures
# =========================
from dataclasses import dataclass

@dataclass
class Position:
    side: Signal
    qty: float
    entry: float
    sl: float
    tp: float
    trail_trigger: float
    be_trigger: float
    peak: float
    notional: float

@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Signal
    entry: float
    exit: float
    qty: float
    pnl: float
    reason: str


# =========================
# Backtester
# =========================
class Backtester:
    def __init__(self, bt_cfg: BacktestConfig, strat_cfg: Optional[StrategyConfig] = None):
        self.bt_cfg = bt_cfg
        self.strat = AlphaComboStrategy(strat_cfg)

    def run(self, candles: pd.DataFrame, initial_capital: float = 1000.0) -> Dict[str, Any]:
        df = self.strat.generate_signals(candles).copy()
        if self.bt_cfg.start:
            df = df[df.index >= pd.to_datetime(self.bt_cfg.start, utc=True)]
        if self.bt_cfg.end:
            df = df[df.index <= pd.to_datetime(self.bt_cfg.end, utc=True)]
        if len(df) == 0:
            log.info("Pas de données après filtres (période/min_bars).")
            return {
                "initial_capital": initial_capital,
                "final_capital": initial_capital,
                "net_profit": 0.0,
                "trades": [],
                "wins": 0,
                "losses": 0,
                "hit_rate": 0.0,
                "max_drawdown": 0.0,
            }

        capital = float(initial_capital)
        pos: Optional[Position] = None
        trades: List[TradeResult] = []
        last_entry_idx: Optional[int] = None
        next_allowed_idx: int = 0  # lockout après perte

        for i in range(1, len(df)):
            t_prev = df.index[i-1]
            t = df.index[i]
            row_prev = df.iloc[i-1]
            row = df.iloc[i]

            # -------- SORTIE / GESTION --------
            if pos is not None:
                atr = max(row["atr"], 1e-9)

                # Break-even: si 1R atteint, on remonte/descend le SL à l'entrée
                if pos.side == "BUY":
                    if row["high"] >= pos.be_trigger and pos.sl < pos.entry:
                        pos.sl = pos.entry
                else:
                    if row["low"] <= pos.be_trigger and pos.sl > pos.entry:
                        pos.sl = pos.entry

                # Trailing dynamique après le trigger
                if pos.side == "BUY":
                    pos.peak = max(pos.peak, row["high"])
                    if row["high"] >= pos.trail_trigger:
                        new_sl = max(pos.sl, pos.peak - self.strat.cfg.trail_atr_mult * atr)
                        pos.sl = max(pos.sl, new_sl)
                else:
                    pos.peak = min(pos.peak, row["low"])
                    if row["low"] <= pos.trail_trigger:
                        new_sl = min(pos.sl, pos.peak + self.strat.cfg.trail_atr_mult * atr)
                        pos.sl = min(pos.sl, new_sl)

                # Conditions de sortie
                exit_reason = None
                exit_price = None
                if pos.side == "BUY":
                    if row["low"] <= pos.sl:
                        exit_reason = "STOP LOSS"
                        exit_price = pos.sl
                    elif row["high"] >= pos.tp:
                        exit_reason = "TAKE PROFIT"
                        exit_price = pos.tp
                else:
                    if row["high"] >= pos.sl:
                        exit_reason = "STOP LOSS"
                        exit_price = pos.sl
                    elif row["low"] <= pos.tp:
                        exit_reason = "TAKE PROFIT"
                        exit_price = pos.tp

                # Reverse signal
                if exit_reason is None and ((row["signal"] == "BUY" and pos.side == "SELL") or (row["signal"] == "SELL" and pos.side == "BUY")):
                    exit_reason = "REVERSE SIGNAL"
                    exit_price = row["close"]

                if exit_reason is not None:
                    raw_exit = apply_slippage(float(exit_price), self.bt_cfg.slippage_bps, pos.side, "exit")
                    notional_exit = raw_exit * pos.qty
                    trade_fees = fees_value(pos.notional, self.bt_cfg.fees_bps) + fees_value(notional_exit, self.bt_cfg.fees_bps)
                    pnl = (raw_exit - pos.entry) * pos.qty if pos.side == "BUY" else (pos.entry - raw_exit) * pos.qty
                    pnl -= trade_fees

                    trades.append(TradeResult(
                        entry_time=t_prev, exit_time=t, side=pos.side, entry=pos.entry, exit=raw_exit,
                        qty=pos.qty, pnl=pnl, reason=exit_reason
                    ))
                    capital += pnl

                    # Lockout si perte
                    if pnl < 0:
                        next_allowed_idx = i + self.bt_cfg.lockout_bars_after_loss

                    log.info(f"[{t}] {exit_reason} @ {raw_exit:.2f} | qty={pos.qty:.6f} | PnL: {pnl:.2f} USDT")
                    pos = None

            # -------- ENTRÉE --------
            if pos is None:
                # Cooldown & lockout
                if i < next_allowed_idx:
                    continue
                if last_entry_idx is not None and (i - last_entry_idx) < self.bt_cfg.cooldown_bars:
                    continue

                sig: Signal = df.iat[i, df.columns.get_loc("signal")]
                if sig not in ("BUY", "SELL"):
                    continue
                if sig == "SELL" and not self.bt_cfg.allow_short:
                    continue

                # Filtre "tradable" (heures/volatilité)
                if "tradable" in df.columns and not bool(row.get("tradable", True)):
                    continue

                atr = max(row["atr"], 1e-9)
                levels = self.strat.compute_levels(sig, float(row["close"]), atr)

                # risque par unité
                per_unit_risk = (row["close"] - levels["sl"]) if sig == "BUY" else (levels["sl"] - row["close"])
                per_unit_risk = max(float(per_unit_risk), 1e-9)

                risk_budget = capital * self.bt_cfg.risk_per_trade
                qty = max((risk_budget * self.bt_cfg.max_leverage) / per_unit_risk, 0.0)
                if qty <= 0:
                    continue

                entry_px = apply_slippage(float(row["close"]), self.bt_cfg.slippage_bps, sig, "entry")
                notional_entry = entry_px * qty
                entry_fees = fees_value(notional_entry, self.bt_cfg.fees_bps)
                capital -= entry_fees

                # break-even trigger
                be_trigger = self.strat.compute_be_trigger(sig, entry_px, levels["sl"])

                pos = Position(
                    side=sig,
                    qty=qty,
                    entry=entry_px,
                    sl=levels["sl"],
                    tp=levels["tp"],
                    trail_trigger=levels["trail_trigger"],
                    be_trigger=be_trigger,
                    peak=float(row["high"] if sig == "BUY" else row["low"]),
                    notional=notional_entry
                )
                last_entry_idx = i

                log.info(
                    f"[{t}] OUVERTURE {('LONG' if sig=='BUY' else 'SHORT')} @ {entry_px:.2f}, SL=@{pos.sl:.2f}, "
                    f"TP=@{pos.tp:.2f} | BE trigger @{pos.be_trigger:.2f} | qty={qty:.6f} BTC"
                )

        # Stats
        pnl_list = [tr.pnl for tr in trades]
        wins = sum(1 for p in pnl_list if p > 0)
        losses = sum(1 for p in pnl_list if p < 0)
        total = len(trades)
        hit = (wins / total * 100.0) if total > 0 else 0.0

        running = initial_capital
        peak = initial_capital
        max_dd = 0.0
        for p in pnl_list:
            running += p
            peak = max(peak, running)
            if peak > 0:
                max_dd = max(max_dd, (peak - running) / peak)

        results = {
            "initial_capital": initial_capital,
            "final_capital": running,
            "net_profit": running - initial_capital,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "hit_rate": hit,
            "max_drawdown": max_dd,
        }
        return results


def main():
    bt_cfg = BacktestConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        risk_per_trade=0.005,   # 0.5% par trade
        max_leverage=1.0,
        fees_bps=6.0,
        slippage_bps=2.0,
        data_csv=None,
        allow_short=True,      # tu peux essayer True plus tard
        start=None,
        end=None,
        cooldown_bars=8,
        lockout_bars_after_loss=10,
    )

    strat_cfg = StrategyConfig(
        ema_fast=21,
        ema_slow=55,
        rsi_len=14,
        rsi_buy=55.0,
        rsi_sell=45.0,
        trend_sma=200,
        atr_len=14,
        atr_mult_sl=2.2,
        atr_mult_tp=3.2,
        trail_atr_mult=1.3,
        min_bars=300,
        adx_len=14,
        adx_min=18.0,
        be_rr=1.0,              # break-even à 1R
        slope_lookback=5,
        trade_start_hour=0,
        trade_end_hour=23,
        avoid_weekends=False,
        atrp_window=200,
        atrp_min_quantile=0.25,
    )

    try:
        candles = load_candles(bt_cfg)
    except Exception as e:
        log.error(f"Erreur chargement données: {e}")
        log.info("Si besoin, fournis un CSV via cfg.data_csv ou laisse le script télécharger via ccxt.")
        return

    log.info("Démarrage du backtest AlphaCombo...")
    engine = Backtester(bt_cfg, strat_cfg)
    res = engine.run(candles, initial_capital=1000.0)

    log.info("===== Résultats du Backtest =====")
    log.info(f"Capital initial : {res['initial_capital']:.2f} USDT")
    log.info(f"Capital final   : {res['final_capital']:.2f} USDT")
    log.info(f"Profit net      : {res['net_profit']:.2f} USDT")
    log.info(f"Trades gagnants : {res['wins']}")
    log.info(f"Trades perdants : {res['losses']}")
    log.info(f"Taux de réussite : {res['hit_rate']:.2f}%")
    log.info(f"Max Drawdown    : {res['max_drawdown']*100:.2f}%")

if __name__ == "__main__":
    main()
