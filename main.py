# main.py
from core.binance_client import BinanceClient
from core.logger import logger
from config import Config
from strategies.sma_rsi_macd import StrategySmaRsiMacd
from core.trade_executor import TradeExecutor


def log_orders(binance):
    """
    Récupère et affiche les ordres ouverts et les derniers historiques
    """
    try:
        open_orders = binance.client.get_open_orders(symbol=Config.TRADING_PAIR)
        logger.info(f"📌 Ordres ouverts sur {Config.TRADING_PAIR} : {open_orders}")

        all_orders = binance.client.get_all_orders(symbol=Config.TRADING_PAIR)
        recent_orders = all_orders[-5:] if len(all_orders) > 5 else all_orders
        logger.info(f"📜 Derniers ordres historiques ({len(recent_orders)}) : {recent_orders}")

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des ordres : {e}")


def main():
    logger.info("Démarrage du bot de trading")
    logger.info(f"Configuration actuelle : {Config.summary()}")

    # --- 1) Connexion Binance
    binance = BinanceClient()

    # --- 2) Récupérer le solde
    balance = binance.get_account_balance("USDT")
    logger.info(f"Solde initial : {balance:.2f} USDT")

    # --- 3) Stratégie
    strategy = StrategySmaRsiMacd()
    klines = binance.get_klines(limit=250)
    signal = strategy.generate_signal(klines)

    logger.info(f"Prix actuel : {signal['price']}")
    logger.info(f"Signal généré : {signal['action']} ({signal['reason']})")

    # --- 4) Exécuter le trade si BUY ou SELL
    if signal['action'] in ["BUY", "SELL"]:
        executor = TradeExecutor()
        result = executor.execute_trade(signal['action'], signal['price'], balance)

        if result['status'] == 'success':
            logger.info("✅ Trade exécuté avec succès sur Binance Testnet")

            # --- 5) Log des ordres ouverts et récents
            log_orders(binance)

        else:
            logger.error(f"Echec du trade : {result['message']}")
    else:
        logger.info("Pas de trade exécuté aujourd'hui (signal HOLD).")


if __name__ == "__main__":
    main()
