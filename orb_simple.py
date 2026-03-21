#!/usr/bin/env python3
"""
ORB 5m Strategy - PROSTA WERSJA
================================
Logika:
1. Opening Range = pierwsza świeca 5m NY (9:30-9:35 ET)
2. Breakout: cena przebija OR High/Low + 0.5 × OR Height
3. Pullback do 0.5 equilibrium impulsu = entry
4. SL na 0.786 fib, TP na -0.1 fib extension
5. Trailing stop: aktywacja po 0.3R, trail 40% zysku
6. Filtr: pre-market momentum (1.5h, prosty kierunek)
7. Max 1 trade/dzień, NY session only

pip install pandas numpy matplotlib kagglehub
python orb_simple.py
"""

import pandas as pd
import numpy as np
from datetime import time as dtime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA
# =============================================================================
def load_data(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Ładowanie: {filepath}")
    else:
        import kagglehub
        path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")
        filepath = os.path.join(path, "1m_data.csv")
    
    df = pd.read_csv(filepath, sep='\t', header=0,
                     names=['DateTime','Open','High','Low','Close','Volume','TickVolume'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M:%S')
    for c in ['Open','High','Low','Close','Volume','TickVolume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Open','High','Low','Close']).sort_values('DateTime').reset_index(drop=True)
    df['date'] = df['DateTime'].dt.date
    df['time'] = df['DateTime'].dt.time
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['dow'] = df['DateTime'].dt.dayofweek
    return df


# =============================================================================
# STRATEGIA
# =============================================================================

# NY session (UTC)
OR_START_H, OR_START_M = 13, 30   # 9:30 ET
OR_END_H, OR_END_M = 13, 35       # 9:35 ET
SESSION_END_H, SESSION_END_M = 20, 0  # 16:00 ET
PM_LOOKBACK_MIN = 90  # 1.5h pre-market


def get_momentum_bias(day_df):
    """Prosty filtr: kierunek pre-marketu 1.5h przed sesją"""
    pm_start = OR_START_H * 60 + OR_START_M - PM_LOOKBACK_MIN
    pm_end = OR_START_H * 60 + OR_START_M
    
    day_min = day_df['hour'] * 60 + day_df['minute']
    pm = day_df[(day_min >= pm_start) & (day_min < pm_end)]
    
    if len(pm) < 5:
        return 'neutral'
    
    change = pm.iloc[-1]['Close'] - pm.iloc[0]['Open']
    rng = pm['High'].max() - pm['Low'].min()
    if rng == 0:
        return 'neutral'
    
    ratio = change / rng
    if ratio > 0.2:
        return 'bull'
    elif ratio < -0.2:
        return 'bear'
    return 'neutral'


def process_day(day_df):
    """Przetworz jeden dzień - zwróć max 1 trade"""
    
    # Opening Range
    or_bars = day_df[(day_df['hour'] == OR_START_H) & 
                     (day_df['minute'] >= OR_START_M) & 
                     (day_df['minute'] < OR_END_M)]
    if len(or_bars) < 2:
        return None
    
    or_high = or_bars['High'].max()
    or_low = or_bars['Low'].min()
    or_height = or_high - or_low
    
    # Filtr: OR height
    if or_height < 5 or or_height > 200:
        return None
    
    session_open = or_bars.iloc[0]['Open']
    
    # Momentum bias
    bias = get_momentum_bias(day_df)
    
    # After-OR data
    or_end = dtime(OR_END_H, OR_END_M)
    session_end = dtime(SESSION_END_H, SESSION_END_M)
    after_or = day_df[(day_df['time'] >= or_end) & (day_df['time'] <= session_end)].reset_index(drop=True)
    
    if len(after_or) < 5:
        return None
    
    # Szukaj breakout
    bull_level = or_high + 0.5 * or_height   # breakout level
    bear_level = or_low - 0.5 * or_height     # breakout level
    
    direction = None
    bo_idx = None
    
    for i in range(len(after_or)):
        if after_or.iloc[i]['High'] >= bull_level:
            direction = 'long'
            bo_idx = i
            break
        if after_or.iloc[i]['Low'] <= bear_level:
            direction = 'short'
            bo_idx = i
            break
    
    if direction is None:
        return None
    
    # Filtr momentum: graj tylko z biasem
    if bias == 'bull' and direction == 'short':
        return None
    if bias == 'bear' and direction == 'long':
        return None
    
    # Impuls + entry
    post_bo = after_or.iloc[bo_idx:].reset_index(drop=True)
    if len(post_bo) < 3:
        return None
    
    if direction == 'long':
        imp_low = session_open
        imp_high = post_bo.iloc[0]['High']
        peak_idx = 0
        
        # Znajdź szczyt impulsu
        for i in range(1, min(60, len(post_bo))):
            if post_bo.iloc[i]['High'] > imp_high:
                imp_high = post_bo.iloc[i]['High']
                peak_idx = i
            elif i > peak_idx:
                rng = imp_high - imp_low
                if rng > 0 and (imp_high - post_bo.iloc[i]['Low']) / rng >= 0.3:
                    break
        
        rng = imp_high - imp_low
        if rng <= 0:
            return None
        
        entry = imp_high - 0.5 * rng        # 0.5 equilibrium
        sl = imp_high - 0.786 * rng          # SL pod 0.786 fib
        tp = imp_high - (-0.1) * rng         # TP = -0.1 fib extension
        risk = entry - sl
        
        if risk <= 0:
            return None
        
        # Szukaj entry (pullback do 0.5)
        for i in range(peak_idx + 1, min(60, len(post_bo))):
            bar = post_bo.iloc[i]
            if bar['time'] > session_end:
                break
            if bar['Low'] <= sl:
                return None  # SL przed entry
            if bar['Low'] <= entry:
                # ENTRY! Symuluj trade
                return simulate(post_bo, i, 'long', entry, sl, tp, risk,
                                session_end, or_high, or_low, or_height,
                                imp_high, imp_low, bias)
    
    else:  # short
        imp_high = session_open
        imp_low = post_bo.iloc[0]['Low']
        trough_idx = 0
        
        for i in range(1, min(60, len(post_bo))):
            if post_bo.iloc[i]['Low'] < imp_low:
                imp_low = post_bo.iloc[i]['Low']
                trough_idx = i
            elif i > trough_idx:
                rng = imp_high - imp_low
                if rng > 0 and (post_bo.iloc[i]['High'] - imp_low) / rng >= 0.3:
                    break
        
        rng = imp_high - imp_low
        if rng <= 0:
            return None
        
        entry = imp_low + 0.5 * rng         # 0.5 equilibrium
        sl = imp_low + 0.786 * rng           # SL nad 0.786 fib
        tp = imp_low + (-0.1) * rng          # TP = -0.1 fib extension
        risk = sl - entry
        
        if risk <= 0:
            return None
        
        for i in range(trough_idx + 1, min(60, len(post_bo))):
            bar = post_bo.iloc[i]
            if bar['time'] > session_end:
                break
            if bar['High'] >= sl:
                return None
            if bar['High'] >= entry:
                return simulate(post_bo, i, 'short', entry, sl, tp, risk,
                                session_end, or_high, or_low, or_height,
                                imp_high, imp_low, bias)
    
    return None


# Trailing stop params
TRAIL_ACTIVATION_RR = 0.3   # aktywacja po 0.3R zysku
TRAIL_DISTANCE_MULT = 0.4   # trail = 40% aktualnego zysku


def simulate(df, entry_idx, direction, entry, sl, tp, risk, session_end,
             or_high, or_low, or_height, imp_high, imp_low, bias):
    """Symuluj trade z trailing stop"""
    entry_bar = df.iloc[entry_idx]
    entry_time = entry_bar['DateTime']
    
    trail_active = False
    trail_sl = sl
    best_price = entry  # najlepsza cena w kierunku trade'a
    
    for i in range(entry_idx + 1, len(df)):
        bar = df.iloc[i]
        
        # Koniec sesji
        if bar['time'] > session_end:
            price = bar['Open']
            pnl = (price - entry) if direction == 'long' else (entry - price)
            exit_type = 'session_end'
            return make_trade(entry_time, bar['DateTime'], direction, entry, price,
                              sl, tp, pnl, exit_type, or_high, or_low, or_height,
                              imp_high, imp_low, risk, i - entry_idx, bias)
        
        current_sl = trail_sl if trail_active else sl
        
        if direction == 'long':
            # Update trailing
            if bar['High'] > best_price:
                best_price = bar['High']
            current_profit = best_price - entry
            if current_profit >= TRAIL_ACTIVATION_RR * risk:
                trail_active = True
                new_trail = entry + current_profit * (1 - TRAIL_DISTANCE_MULT)
                trail_sl = max(trail_sl, new_trail)
                current_sl = trail_sl
            
            if bar['Low'] <= current_sl:
                pnl = current_sl - entry
                exit_type = 'trail_sl' if trail_active else 'sl'
                return make_trade(entry_time, bar['DateTime'], direction, entry, current_sl,
                                  sl, tp, pnl, exit_type, or_high, or_low, or_height,
                                  imp_high, imp_low, risk, i - entry_idx, bias)
            if bar['High'] >= tp:
                pnl = tp - entry
                return make_trade(entry_time, bar['DateTime'], direction, entry, tp,
                                  sl, tp, pnl, 'tp', or_high, or_low, or_height,
                                  imp_high, imp_low, risk, i - entry_idx, bias)
        else:
            if bar['Low'] < best_price:
                best_price = bar['Low']
            current_profit = entry - best_price
            if current_profit >= TRAIL_ACTIVATION_RR * risk:
                trail_active = True
                new_trail = entry - current_profit * (1 - TRAIL_DISTANCE_MULT)
                trail_sl = min(trail_sl, new_trail)
                current_sl = trail_sl
            
            if bar['High'] >= current_sl:
                pnl = entry - current_sl
                exit_type = 'trail_sl' if trail_active else 'sl'
                return make_trade(entry_time, bar['DateTime'], direction, entry, current_sl,
                                  sl, tp, pnl, exit_type, or_high, or_low, or_height,
                                  imp_high, imp_low, risk, i - entry_idx, bias)
            if bar['Low'] <= tp:
                pnl = entry - tp
                return make_trade(entry_time, bar['DateTime'], direction, entry, tp,
                                  sl, tp, pnl, 'tp', or_high, or_low, or_height,
                                  imp_high, imp_low, risk, i - entry_idx, bias)
    
    last = df.iloc[-1]
    pnl = (last['Close'] - entry) if direction == 'long' else (entry - last['Close'])
    return make_trade(entry_time, last['DateTime'], direction, entry, last['Close'],
                      sl, tp, pnl, 'eod', or_high, or_low, or_height,
                      imp_high, imp_low, risk, len(df) - entry_idx, bias)


def make_trade(entry_time, exit_time, direction, entry, exit_price, sl, tp, pnl,
               exit_type, or_high, or_low, or_height, imp_high, imp_low, risk, bars, bias):
    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'date': entry_time.date(),
        'direction': direction,
        'entry_price': round(entry, 2),
        'exit_price': round(exit_price, 2),
        'sl': round(sl, 2),
        'tp': round(tp, 2),
        'pnl_points': round(pnl, 2),
        'exit_type': exit_type,
        'or_high': round(or_high, 2),
        'or_low': round(or_low, 2),
        'or_height': round(or_height, 2),
        'impulse_high': round(imp_high, 2),
        'impulse_low': round(imp_low, 2),
        'risk': round(risk, 2),
        'rr': round(pnl / risk, 2) if risk != 0 else 0,
        'bars_held': bars,
        'bias': bias,
        'win': pnl > 0,
    }


# =============================================================================
# BACKTEST
# =============================================================================
def run_backtest(df, start_date='2024-01-01'):
    df = df[df['DateTime'] >= start_date].reset_index(drop=True)
    grouped = dict(list(df.groupby('date')))
    dates = sorted(grouped.keys())
    
    trades = []
    for idx, date in enumerate(dates):
        if idx % 100 == 0:
            print(f"  [{idx+1}/{len(dates)}] {date}...")
        
        day_df = grouped[date].reset_index(drop=True)
        trade = process_day(day_df)
        if trade:
            trades.append(trade)
    
    return pd.DataFrame(trades)


# =============================================================================
# STATYSTYKI
# =============================================================================
def compute_stats(t):
    if len(t) == 0:
        return {'total_trades': 0}
    
    s = {}
    s['total_trades'] = len(t)
    s['winners'] = int(t['win'].sum())
    s['losers'] = s['total_trades'] - s['winners']
    s['win_rate'] = round(s['winners'] / s['total_trades'] * 100, 1)
    
    s['total_pnl'] = round(t['pnl_points'].sum(), 1)
    s['avg_pnl'] = round(t['pnl_points'].mean(), 2)
    s['avg_winner'] = round(t.loc[t['win'], 'pnl_points'].mean(), 2) if s['winners'] else 0
    s['avg_loser'] = round(t.loc[~t['win'], 'pnl_points'].mean(), 2) if s['losers'] else 0
    s['max_win'] = round(t['pnl_points'].max(), 2)
    s['max_loss'] = round(t['pnl_points'].min(), 2)
    
    gp = t.loc[t['win'], 'pnl_points'].sum()
    gl = abs(t.loc[~t['win'], 'pnl_points'].sum())
    s['profit_factor'] = round(gp / gl, 2) if gl > 0 else float('inf')
    
    s['avg_rr'] = round(t['rr'].mean(), 2)
    
    daily = t.groupby('date')['pnl_points'].sum()
    s['trading_days'] = len(daily)
    s['green_days'] = int((daily > 0).sum())
    s['red_days'] = int((daily <= 0).sum())
    s['green_day_pct'] = round(s['green_days'] / s['trading_days'] * 100, 1)
    s['avg_green_day'] = round(daily[daily > 0].mean(), 1) if s['green_days'] else 0
    s['avg_red_day'] = round(daily[daily <= 0].mean(), 1) if s['red_days'] else 0
    
    tc = t.copy()
    tc['month'] = pd.to_datetime(tc['entry_time']).dt.to_period('M')
    monthly = tc.groupby('month')['pnl_points'].sum()
    s['total_months'] = len(monthly)
    s['green_months'] = int((monthly > 0).sum())
    s['green_month_pct'] = round(s['green_months'] / s['total_months'] * 100, 1)
    
    cum = t['pnl_points'].cumsum()
    dd = cum - cum.cummax()
    s['max_dd'] = round(dd.min(), 1)
    
    s['tp_exits'] = int((t['exit_type'] == 'tp').sum())
    s['sl_exits'] = int((t['exit_type'] == 'sl').sum())
    s['session_exits'] = int((t['exit_type'] == 'session_end').sum())
    s['trail_exits'] = int((t['exit_type'] == 'trail_sl').sum())
    
    for d in ['long', 'short']:
        dt = t[t['direction'] == d]
        if len(dt):
            s[f'{d}_trades'] = len(dt)
            s[f'{d}_wr'] = round(dt['win'].sum() / len(dt) * 100, 1)
            s[f'{d}_pnl'] = round(dt['pnl_points'].sum(), 1)
    
    # Streaks
    wins = t['win'].values
    max_ws = max_ls = cw = cl = 0
    for w in wins:
        if w:
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        max_ws = max(max_ws, cw)
        max_ls = max(max_ls, cl)
    s['max_win_streak'] = max_ws
    s['max_loss_streak'] = max_ls
    s['avg_bars'] = round(t['bars_held'].mean(), 1)
    s['expectancy'] = round((s['win_rate']/100 * s['avg_winner']) + ((1 - s['win_rate']/100) * s['avg_loser']), 2)
    
    return s


def print_report(s):
    print(f"""
{'='*55}
  ORB 5m - Prosta Strategia (NY Only + Momentum + Trail)
{'='*55}

  TRADES
  ─────────────────────────────────
  Total:          {s['total_trades']:>5}
  Winners:        {s['winners']:>5}   ({s['win_rate']}%)
  Losers:         {s['losers']:>5}
  Trades/Day:     {s['total_trades']/s['trading_days']:.2f}

  P&L
  ─────────────────────────────────
  Total PnL:    {s['total_pnl']:>8.1f} pts
  Avg PnL:      {s['avg_pnl']:>8.2f} pts
  Avg Winner:   {s['avg_winner']:>8.2f} pts
  Avg Loser:    {s['avg_loser']:>8.2f} pts
  Max Win:      {s['max_win']:>8.2f} pts
  Max Loss:     {s['max_loss']:>8.2f} pts
  Profit Factor:{s['profit_factor']:>8.2f}
  Expectancy:   {s['expectancy']:>8.2f} pts

  GREEN DAYS
  ─────────────────────────────────
  Trading Days:   {s['trading_days']:>5}
  Green Days:     {s['green_days']:>5}  ({s['green_day_pct']}%)
  Red Days:       {s['red_days']:>5}
  Avg Green Day:  {s['avg_green_day']:>7.1f} pts
  Avg Red Day:    {s['avg_red_day']:>7.1f} pts
  Green Months:   {s['green_months']}/{s['total_months']}  ({s['green_month_pct']}%)

  DRAWDOWN & STREAKS
  ─────────────────────────────────
  Max Drawdown:   {s['max_dd']:>7.1f} pts
  Max Win Streak: {s['max_win_streak']:>5}
  Max Loss Streak:{s['max_loss_streak']:>5}
  Avg Bars Held:  {s['avg_bars']:>5}

  EXITS
  ─────────────────────────────────
  TP:       {s['tp_exits']:>5}
  SL:       {s['sl_exits']:>5}
  Trail SL: {s.get('trail_exits', 0):>5}
  Session:  {s['session_exits']:>5}

  DIRECTION
  ─────────────────────────────────""")
    for d in ['long', 'short']:
        if f'{d}_trades' in s:
            print(f"  {d.upper():>7}: {s[f'{d}_trades']:>4} trades, WR={s[f'{d}_wr']}%, PnL={s[f'{d}_pnl']:>7.1f}")
    print(f"{'='*55}\n")


# =============================================================================
# WYKRESY
# =============================================================================
def plot_equity(trades_df, stats, save_path='/home/ubuntu/orb_simple_equity.png'):
    t = trades_df.copy()
    t['cum_pnl'] = t['pnl_points'].cumsum()
    t['dt'] = pd.to_datetime(t['entry_time'])
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('ORB 5m Strategy - Prosta wersja (NY Only, Momentum + Trail)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t['dt'], t['cum_pnl'], color='#1976D2', linewidth=1.5)
    ax1.fill_between(t['dt'], 0, t['cum_pnl'], where=t['cum_pnl']>=0, color='#4CAF50', alpha=0.15)
    ax1.fill_between(t['dt'], 0, t['cum_pnl'], where=t['cum_pnl']<0, color='#F44336', alpha=0.15)
    ax1.axhline(0, color='gray', ls='--', alpha=0.5)
    ax1.set_title('Equity Curve (cumulative PnL)', fontsize=12)
    ax1.set_ylabel('Points')
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # DD shading
    cum_max = t['cum_pnl'].cummax()
    dd = t['cum_pnl'] - cum_max
    ax1b = ax1.twinx()
    ax1b.fill_between(t['dt'], dd, 0, color='red', alpha=0.08)
    ax1b.set_ylabel('Drawdown', color='red', alpha=0.5)
    
    # 2. Daily PnL
    daily = t.groupby('date')['pnl_points'].sum()
    ax2 = fig.add_subplot(gs[1, 0])
    cols = ['#4CAF50' if x > 0 else '#F44336' for x in daily.values]
    ax2.bar(range(len(daily)), daily.values, color=cols, alpha=0.7, width=1)
    ax2.axhline(0, color='gray', ls='-', alpha=0.5)
    ax2.set_title(f'Daily PnL ({stats["green_day_pct"]}% green)', fontsize=11)
    ax2.set_ylabel('Points')
    ax2.set_xlabel('Trading Day')
    ax2.grid(alpha=0.3)
    
    # 3. Monthly PnL
    tc = t.copy()
    tc['month'] = pd.to_datetime(tc['entry_time']).dt.to_period('M')
    monthly = tc.groupby('month')['pnl_points'].sum()
    ax3 = fig.add_subplot(gs[1, 1])
    mc = ['#4CAF50' if x > 0 else '#F44336' for x in monthly.values]
    ax3.bar(range(len(monthly)), monthly.values, color=mc, alpha=0.7)
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=7)
    ax3.axhline(0, color='gray', ls='-', alpha=0.5)
    ax3.set_title(f'Monthly PnL ({stats["green_months"]}/{stats["total_months"]} green)', fontsize=11)
    ax3.set_ylabel('Points')
    ax3.grid(alpha=0.3)
    
    # 4. R:R histogram
    ax4 = fig.add_subplot(gs[2, 0])
    rr = t['rr'].clip(-3, 3)
    ax4.hist(rr, bins=30, color='#1976D2', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='red', ls='--', alpha=0.7)
    ax4.axvline(rr.median(), color='green', ls='--', alpha=0.7, label=f'Median={rr.median():.2f}')
    ax4.set_title('R:R Distribution', fontsize=11)
    ax4.set_xlabel('R:R')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    txt = (
        f"PODSUMOWANIE\n"
        f"{'─'*28}\n"
        f"Trades:       {stats['total_trades']}\n"
        f"Win Rate:     {stats['win_rate']}%\n"
        f"Green Days:   {stats['green_day_pct']}%\n"
        f"Green Months: {stats['green_months']}/{stats['total_months']}\n"
        f"Profit Factor:{stats['profit_factor']}\n"
        f"Total PnL:    {stats['total_pnl']} pts\n"
        f"Max Drawdown: {stats['max_dd']} pts\n"
        f"Avg Winner:   {stats['avg_winner']} pts\n"
        f"Avg Loser:    {stats['avg_loser']} pts\n"
        f"Expectancy:   {stats['expectancy']} pts\n"
        f"{'─'*28}\n"
        f"Avg R:R:      {stats['avg_rr']}\n"
        f"Win Streak:   {stats['max_win_streak']}\n"
        f"Loss Streak:  {stats['max_loss_streak']}\n"
    )
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes, fontsize=10,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Wykres: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  ORB 5m - Uproszczona Strategia NQ")
    print("=" * 55)
    
    data_path = '/home/ubuntu/.cache/kagglehub/datasets/novandraanugrah/nasdaq-100-nas100-historical-price-data/versions/3/1m_data.csv'
    df = load_data(data_path)
    df = df[df['DateTime'] >= '2024-01-01'].reset_index(drop=True)
    print(f"Dane: {len(df)} świec | {df['DateTime'].min()} → {df['DateTime'].max()}")
    
    trades = run_backtest(df, '2024-01-01')
    print(f"\nZnaleziono {len(trades)} tradów")
    
    stats = compute_stats(trades)
    print_report(stats)
    
    plot_equity(trades, stats, '/home/ubuntu/orb_simple_equity.png')
    trades.to_csv('/home/ubuntu/orb_simple_trades.csv', index=False)
    
    print("Pliki:")
    print("  orb_simple.py           - strategia")
    print("  orb_simple_equity.png   - equity curve")
    print("  orb_simple_trades.csv   - trade log")
