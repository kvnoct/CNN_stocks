import pandas as pd
import numpy as np

def get_rsi(df, intervals):
    """
        take max(intervals) data
    """
    prev_average_gain = 0
    prev_average_loss = 0
    
    def calculate_rsi(data):
        nonlocal rolling
        nonlocal prev_average_gain
        nonlocal prev_average_loss
        
        
        current_gain = data.where(data >= 0, 0)
        current_loss = np.abs(data.where(data < 0, 0))
        
        #print(rolling)
        
    
        if(rolling == 0):    
            average_gain = current_gain.sum() / period
            average_loss = current_loss.sum() / period
            a = average_gain / average_loss
            rsi = 100 - (100/ ( 1 + a ))
            #print(average_gain)
        else:
            #print(current_gain.iloc[-1])
            
            average_gain = ((prev_average_gain * (period - 1)) + current_gain.iloc[-1]) / period
            average_loss = ((prev_average_loss * (period - 1)) + current_loss.iloc[-1]) / period
            a = average_gain / average_loss
            #print(average_gain)
            rsi = 100 - (100/ ( 1 + a ))

        prev_average_gain = average_gain
        prev_average_loss = average_loss

        rolling = rolling + 1
        return rsi
    
    change = df['Close'].diff()[1:]

    for period in intervals:
        df[f'rsi_{period}'] = np.nan
        rolling = 0
        res = change.rolling(period).apply(calculate_rsi, raw = False)
        #print(res)
        df[f'rsi_{period}'][1:] = res


def get_williamR(df,intervals):
    """
        take max(intervals) - 1 data
    """

    for period in intervals:
        df[f'wR_{period}'] = np.nan
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        
        for i in range(period-1, df.shape[0]):
            if (high_max[i] == low_min[i] ):
                df[f'wR_{period}'][i] = 0
            else:
                a = -100 * ( high_max[i] - df['Close'][i] ) / (high_max[i] - low_min[i] )
                df[f'wR_{period}'][i] = a
            
        
        #df[f'wR_{period}'].fillna(0)

def get_sma(df, intervals):
    """
        take max(intervals) - 1 data
    """
    for period in intervals:
        df[f'sma_{period}'] = np.nan
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
    

def get_ema(df,intervals, tema = 0, macd = 0, ppo = 0): 
    """
        take max(intervals) - 1 data
    """

    prev_ema = 0
    rolling = 0
    
    def calculate_ema(data, period):
        nonlocal rolling
        nonlocal prev_ema
        #print(data)
        #print(data.iloc[-1])
        #print(rolling)
        #print(period)
        if rolling == 0:
            ema = data.mean()
        else:
            smoothing_const = 2 / (1 + period)
            #print(smoothing_const)
            ema = (smoothing_const * ( data.iloc[-1] - prev_ema )) + prev_ema
        
        
        prev_ema = ema
        rolling += 1
        
        return ema
    
    for period in intervals:
        if(tema == 0 and macd == 0):
            df[f'ema_{period}'] = np.nan
            rolling = 0
            res = df['Close'].rolling(period).apply(calculate_ema, args=(period,))

            df[f'ema_{period}'] = res
        elif(tema == 1):
            df[f'ema_ema_{period}'] = np.nan
            rolling = 0
            
            res = df[f'ema_{period}'].rolling(period).apply(calculate_ema, args=(period,))

            df[f'ema_ema_{period}'] = res
        elif(tema == 2):
            df[f'ema_ema_ema_{period}'] = np.nan
            rolling = 0
            
            res = df[f'ema_ema_{period}'].rolling(period).apply(calculate_ema, args=(period,))

            df[f'ema_ema_ema_{period}'] = res
        if(macd == 1):
            df['macd_signal'] = np.nan
            rolling = 0
            res = df['Close'].rolling(period).apply(calculate_ema, args=(period,))

            df['macd_signal'] = res
        if(ppo == 1):
            df['ppo_signal'] = np.nan
            rolling = 0
            res = df['Close'].rolling(period).apply(calculate_ema, args=(period,))

            df['ppo_signal'] = res
        
def get_wma(df, intervals, hma_calc = 0):
    def calculate_wma(data, period):
        weights = pd.Series(range(1, period+1))
        return np.multiply(data.values, weights.values).sum() / weights.sum()
    
    for period in intervals:
        if(hma_calc == 0):
            df[f"wma_{period}"] = np.nan

            res = df['Price'].rolling(period).apply(calculate_wma, args=(period,))
            df[f'wma_{period}'] = res
        else:
            n = np.round(period**2).astype(int)
            df[f"hma_{n}"] = np.nan
            
            res = df[f'hma_tmp_{n}'].rolling(period).apply(calculate_wma, args=(period,))
            df[f"hma_{n}"] = res

#not implemented    
def get_hma(df, intervals):
    """
        intervals must be perfect square and divisible by 2
    """
    
    #STEP 1: WMA(period/2)
    half_intervals = np.round( [i/2 for i in intervals] ).astype(int)
    get_wma(df, half_intervals)
    
    #STEP 2: 2*STEP_1 - WMA(period)
    get_wma(df, intervals)
    
    for period in intervals:
        df[f'hma_tmp_{period}'] = df[f'wma_{np.round(period/2).astype(int)}'] * 2 - df[f'wma_{period}']
        
    #print(df)
    #STEP 3: WMA(M = STEP_2, period = sqrt(n))
    sqrt_period = np.round( [np.sqrt(i) for i in intervals]).astype(int)
    get_wma(df, sqrt_period, 1)
    
    for period in intervals:
        df = df.drop(columns = [f'wma_{np.round(period/2).astype(int)}'], axis = 1)
        df = df.drop(columns = [f'wma_{period}'], axis = 1)
        df = df.drop(columns = [f'hma_tmp_{period}'], axis = 1)
    
    return df
    
def get_tema(df, intervals, keep_ema = 1):
    """
        take 3 * period - 2 data
        
        arg = 
        keep_ema -> 1 if you want to keep ema indicators in the dataframes (default) 
                    0 otherwise

        return: new dataframe
    """
    ema_flag = []

    for period in intervals:
        if("ema_{period}") in df:
            ema_flag.append(period)


    #STEP 1: EMA(EMA)
    get_ema(df, intervals) #get EMA
    get_ema(df,intervals, 1) #get EMA(EMA)
    
    #STEP 2: EMA(EMA(EMA))
    get_ema(df,intervals, 2) #EMA(EMA(EMA))
    
    #STEP 3: ( 3*EMA - 3*STEP_1 ) + STEP_3
    for period in intervals:
        df[f'tema_{period}'] = ( 3*df[f'ema_{period}'] - 3*df[f'ema_ema_{period}'] ) + df[f'ema_ema_ema_{period}']
    
    for period in intervals:
        if keep_ema == 0:
            del df[f'ema_{period}']

        del df[f'ema_ema_{period}']
        del df[f'ema_ema_ema_{period}']
        #df = df.drop(columns = [f'ema_ema_{period}'], axis = 1)
        #df = df.drop(columns = [f'ema_ema_ema_{period}'], axis = 1)
    
    
    return df

def get_cci(df, intervals):
    """
        take max(intervals) - 1 data
    """
    mad = lambda x: np.fabs(x- x.mean()).mean()
    
    for period in intervals:
        typical_price = ( df['High'] + df['Close'] + df['Low'] ) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_dev = typical_price.rolling(period).apply(mad)
        
        df[f'cci_{period}'] = ( typical_price - sma_tp ) / ( 0.015 * mean_dev)

def get_cmo(df, intervals):
    """
        take max(intervals) data
    """ 
    def calculate_cmo(data, period):
        gains = data[data >= 0].sum()
        loss  = np.abs(data[data < 0].sum())
        if(gains+loss == 0):
            cmo = 0
        else:
            cmo = 100 * ((gains - loss) / (gains + loss))
        return cmo
    
    change = df['Close'].diff()[1:]
    
    for period in intervals:
        df[f'cmo_{period}'] = np.nan
        res = change.rolling(period).apply(calculate_cmo, args=(period,))
        df[f'cmo_{period}'] = res

#How it can be used for another time periods ?
def get_macd(df):
    flag_12 = 0; flag_26 = 0
    
    if('ema_12' in df):
        flag_12 = 1
    else:
        get_ema(df, [12])
        
    if('ema_26' in df):
        flag_26 = 1
    else:
        get_ema(df, [26])
    
    macd_line = df['ema_12'] - df['ema_26']
    get_ema(df, [9], macd = 1)
    
    if(flag_12 == 0):
        df.drop(column = ['ema_12'], axis = 1)
    if(flag_26 == 0):
        df.drop(column = ['ema_26'], axis = 1)

#How it can be used for another time periods ?
def get_ppo(df):
    flag_12 = 0; flag_26 = 0
    
    if('ema_12' in df):
        flag_12 = 1
    else:
        get_ema(df, [12])
        
    if('ema_26' in df):
        flag_26 = 1
    else:
        get_ema(df, [26])
    
    ppo_line = 100 * ( df['ema_12'] - df['ema_26'] ) / df['ema_26']
    get_ema(df, [9], ppo = 1)
    
    if(flag_12 == 0):
        df.drop(column = ['ema_12'], axis = 1)
    if(flag_26 == 0):
        df.drop(column = ['ema_26'], axis = 1)

def get_roc(df, intervals):
    """
        take max(intervals) data
    """
    def calculate_roc(data, period):
        roc = 100 * (data.iloc[-1] - data.iloc[0]) / data.iloc[0] 
        return roc
    
    for period in intervals:
        df[f'roc_{period}'] = np.nan
        res = df['Close'].rolling(period+1).apply(calculate_roc, args=(period,))
        df[f'roc_{period}'] = res

def get_cmf(df, intervals):
    """
        take max(intervals) - 1 data
    """
    multiplier = ( ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) ).fillna(0)
    volume = df['Volume']
    money_flow_volume = multiplier * volume

    for period in intervals:
        df[f'cmf_{period}'] = np.nan
    
        money_flow_volume_sum = money_flow_volume.rolling(period).sum()
        volume_sum = volume.rolling(period).sum()
        
        cmf = money_flow_volume_sum / volume_sum
        
        df[f'cmf_{period}'] = cmf

def get_adx(df, intervals):
    """
        take 2 * max(intervals) - 1 data
    """

    tr = []; dm_plus = []; dm_neg = []
    rolling = 0
    prev_smooth = 0
    prev_adx = 0
    
    def calculate_smooth(data, period):
        nonlocal rolling
        nonlocal prev_smooth
        
        if rolling == 0:
            smooth = data.sum()
        else:
            smooth = prev_smooth - (prev_smooth / period) + data.iloc[-1]
        
        rolling += 1   
        prev_smooth = smooth
        return smooth

    def calculate_adx(data, period):
        nonlocal rolling
        nonlocal prev_adx
    
        if rolling == 0:
            adx = data.mean()
        else:
            adx = (( prev_adx * (period - 1) ) + data.iloc[-1] ) / period
         
        rolling += 1
        prev_adx = adx
        return adx
    
    for i in range(len(df)):
        if i == 0:
            tr.append(None)
            dm_plus.append(None)
            dm_neg.append(None)
        else:        
            high = df['High'][i-1:i+1]
            low = df['Low'][i-1:i+1]
            close = df['Close'][i-1:i+1]
            
            #Calculate TR
            ch_cl = high.iloc[-1] - low.iloc[-1] #current high - current low
            ch_pc = high.iloc[-1] - close.iloc[0] #current high - previous low
            cl_pc = low.iloc[-1] - close.iloc[0] #current low - previous close
            choose_list = np.abs([ch_cl, ch_pc, cl_pc])
            tr.append(max(choose_list))
            
            #Calculate +DM and -DM
            ch_ph = high.iloc[-1] - high.iloc[0] #current high - previous high
            pl_cl = low.iloc[0] - low.iloc[-1] #previous low - current low
            
            if(ch_ph > pl_cl and ch_ph > 0):
                dm_plus.append(ch_ph)
                dm_neg.append(0)
            elif(pl_cl > ch_ph and pl_cl > 0):
                dm_neg.append(pl_cl)
                dm_plus.append(0)
            else:
                dm_plus.append(0)
                dm_neg.append(0)
            
        tr_series = pd.Series(tr)
        dm_plus_series = pd.Series(dm_plus)
        dm_neg_series = pd.Series(dm_neg)
        
    for period in intervals:
        df[f'adx_{period}'] = np.nan
        
        rolling = 0
        tr_smooth = tr_series.rolling(period).apply(calculate_smooth, args=(period,))
        rolling = 0
        dm_plus_smooth = dm_plus_series.rolling(period).apply(calculate_smooth, args=(period,))
        rolling = 0
        dm_neg_smooth = dm_neg_series.rolling(period).apply(calculate_smooth, args=(period,))
        
        di_plus_series = 100 * (dm_plus_smooth / tr_smooth)
        di_neg_series = 100 * (dm_neg_smooth / tr_smooth)
        di_diff = np.abs(di_plus_series - di_neg_series)
        di_sum = di_plus_series + di_neg_series
        dx = 100 * (di_diff / di_sum)
       
        rolling = 0
        adx = dx.rolling(period).apply(calculate_adx, args=(period,))
        df[f'adx_{period}'] = adx
        