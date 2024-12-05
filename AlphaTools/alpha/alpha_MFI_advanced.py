import pandas as pd
from ta.momentum import ROCIndicator
from ta.volatility import AverageTrueRange
from ta.volume import MFIIndicator
from ..base.alpha import BaseAlpha
from ..utils.filters import lowpass_filter


class EnhancedMFIAlpha(BaseAlpha):
    def __init__(self,
                 stock_data,
                 expiration_date,
                 mfi_period=14,
                 mfi_upper=80,
                 mfi_lower=20,
                 roc_period=10,
                 ma_short=20,
                 ma_long=50,
                 atr_period=14,
                 atr_multiplier=2):
        super().__init__(stock_data, expiration_date)
        # MFI parameters
        self.mfi_period = mfi_period
        self.mfi_upper = mfi_upper
        self.mfi_lower = mfi_lower
        # Thêm các tham số mới
        self.roc_period = roc_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        # 1. Moving Averages để xác định trend
        df['ma_short'] = df['Close'].rolling(self.ma_short).mean()
        df['ma_long'] = df['Close'].rolling(self.ma_long).mean()
        
        # 2. MFI
        mfi = MFIIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=self.mfi_period
        )
        df['mfi'] = mfi.money_flow_index()
        
        # 3. Rate of Change - momentum indicator
        df['roc'] = ROCIndicator(
            close=df['Close'],
            window=self.roc_period
        ).roc()
        
        # 4. ATR cho stop loss động
        atr = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.atr_period
        )
        df['atr'] = atr.average_true_range()
        
        # 5. Volume trend
        df['volume_ma'] = df['Volume'].rolling(self.ma_short).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        return df

    def generate_signals(self, df, i, current_position) -> int:
        """Generate enhanced trading signals"""
        # Kiểm tra trend
        uptrend = df['ma_short'].iloc[i] > df['ma_long'].iloc[i]
        downtrend = df['ma_short'].iloc[i] < df['ma_long'].iloc[i]
        
        # Volume confirmation
        strong_volume = df['volume_ratio'].iloc[i] > 1.2
        
        # Momentum confirmation
        positive_momentum = df['roc'].iloc[i] > 0
        negative_momentum = df['roc'].iloc[i] < 0
        
        # Dynamic stop loss
        if current_position != 0:
            stop_loss = self.check_stop_loss(df, i, current_position)
            if stop_loss:
                return 0

        # Signal generation
        if current_position == 0:  # Không có position
            if (df['mfi'].iloc[i] < self.mfi_lower and  # MFI thấp
                uptrend and                             # Trong uptrend
                positive_momentum and                   # Momentum tốt
                strong_volume):                         # Volume mạnh
                return 1  # Long signal
                
            elif (df['mfi'].iloc[i] > self.mfi_upper and  # MFI cao
                  downtrend and                           # Trong downtrend
                  negative_momentum and                   # Momentum xấu
                  strong_volume):                         # Volume mạnh
                return -1  # Short signal
                
        # Exit conditions
        elif current_position == 1:  # Đang long
            if (df['mfi'].iloc[i] > 70 or      # MFI quá cao
                not uptrend or                  # Mất uptrend
                not positive_momentum):         # Mất momentum
                return 0  # Exit long
                
        elif current_position == -1:  # Đang short
            if (df['mfi'].iloc[i] < 30 or      # MFI quá thấp
                not downtrend or               # Mất downtrend
                not negative_momentum):        # Mất momentum
                return 0  # Exit short
        
        return current_position

    def check_stop_loss(self, df, i, position):
        """Check dynamic stop loss"""
        if i < 1:  # Không đủ data
            return False
            
        if position == 1:  # Long position
            stop_price = df['Close'].iloc[i-1] - self.atr_multiplier * df['atr'].iloc[i-1]
            if df['Low'].iloc[i] < stop_price:
                return True
                
        elif position == -1:  # Short position
            stop_price = df['Close'].iloc[i-1] + self.atr_multiplier * df['atr'].iloc[i-1]
            if df['High'].iloc[i] > stop_price:
                return True
                
        return False