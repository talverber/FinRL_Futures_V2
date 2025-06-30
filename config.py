"""Global configuration for training and backtesting scripts."""

# Data folder to use for training and backtesting
# Possible options include 'futures_data' or 'retail_data'
DATA_TYPE = 'futures_data'

if DATA_TYPE == 'retail_data':
    INDICATORS = ['rmean_7', 'rmean_30', 'vix']

elif DATA_TYPE == 'futures_data':
    INDICATORS = ['ret','ret_1M','ret_2M','ret_3M','ret_1Y','macd', 'rsi_30']

elif DATA_TYPE == 'reg_data':
    from finrl.config import INDICATORS as STOCK_INDICATORS
    INDICATORS = STOCK_INDICATORS #['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
# not what we need for futures
else:
    raise NotImplementedError("unknown DATA_TYPE: " + DATA_TYPE)



