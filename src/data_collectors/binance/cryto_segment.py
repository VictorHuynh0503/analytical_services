import pandas as pd

# Example: your existing Binance snapshot DataFrame
# df = your_dataframe_from_binance()

def generate_ticker_symbol():
# --- Define mapping dictionary (symbol: category info) ---
    project_info = {
        "BTCUSDT": {"chain": "Bitcoin", "theme": "Store of Value", "type": "Layer-1"},
        "ETHUSDT": {"chain": "Ethereum", "theme": "Smart Contracts / DeFi / NFT", "type": "Layer-1"},
        "SOLUSDT": {"chain": "Solana", "theme": "Smart Contracts / DeFi / NFT", "type": "Layer-1"},
        "BNBUSDT": {"chain": "BNB Chain", "theme": "Smart Contracts / DeFi", "type": "Layer-1"},
        "MANAUSDT": {"chain": "Ethereum", "theme": "Metaverse / NFT", "type": "Gaming"},
        "SANDUSDT": {"chain": "Ethereum", "theme": "Metaverse / NFT", "type": "Gaming"},
        "APEUSDT": {"chain": "Ethereum", "theme": "NFT / DAO", "type": "NFT"},
        "IMXUSDT": {"chain": "Ethereum (Layer-2)", "theme": "NFT / Gaming", "type": "Layer-2"},
        "FLOWUSDT": {"chain": "Flow", "theme": "NFT / Gaming", "type": "Layer-1"},
        "GRTUSDT": {"chain": "Ethereum", "theme": "Indexing / Infrastructure", "type": "Middleware"},
        "VETUSDT": {"chain": "VeChain", "theme": "Supply Chain / Enterprise", "type": "Layer-1"},
        "THETAUSDT": {"chain": "Theta", "theme": "Video Streaming / NFT", "type": "Layer-1"},
        "AUDIOUSDT": {"chain": "Ethereum", "theme": "Music / NFT", "type": "App Token"},
        "ORDIUSDT": {"chain": "Bitcoin", "theme": "Ordinals / NFT", "type": "Layer-1 Ext"}
    }

    # --- Convert mapping into DataFrame ---
    project_df = pd.DataFrame.from_dict(project_info, orient="index").reset_index()
    project_df.rename(columns={"index": "symbol"}, inplace=True)

    return project_df