"""dashboard.py

Streamlit Dashboard for monitoring RL Portfolio Paper Trading.

Usage:
    streamlit run dashboard.py

Features:
- Real-time portfolio value
- Current positions
- Trade history
- Performance charts
- Comparison vs SPY
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Alpaca API
from alpaca_trade_api import REST

# Import configuration
try:
    from config import (
        ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, STOCK_TICKERS
    )
except ImportError:
    st.error("[ERROR] config.py not found! Please create it with your API credentials.")
    st.stop()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="RL Portfolio Dashboard",
    page_icon="chart",
    layout="wide"
)

# =============================================================================
# ALPACA CONNECTION
# =============================================================================

@st.cache_resource
def get_alpaca_client():
    """Create Alpaca API client."""
    return REST(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        base_url=ALPACA_BASE_URL
    )

# =============================================================================
# DATA FETCHING
# =============================================================================

def get_account_info(api):
    """Fetch account information."""
    account = api.get_account()
    return {
        'portfolio_value': float(account.portfolio_value),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power),
        'equity': float(account.equity),
        'last_equity': float(account.last_equity),
        'daily_pnl': float(account.equity) - float(account.last_equity),
        'status': account.status
    }

def get_positions(api):
    """Fetch current positions."""
    positions = api.list_positions()
    if not positions:
        return pd.DataFrame()
    
    data = []
    for pos in positions:
        data.append({
            'Symbol': pos.symbol,
            'Qty': int(float(pos.qty)),
            'Avg Cost': float(pos.avg_entry_price),
            'Current Price': float(pos.current_price),
            'Market Value': float(pos.market_value),
            'P&L': float(pos.unrealized_pl),
            'P&L %': float(pos.unrealized_plpc) * 100
        })
    
    return pd.DataFrame(data)

def get_portfolio_history(api, days=30):
    """Fetch portfolio value history."""
    try:
        history = api.get_portfolio_history(
            period=f"{days}D",
            timeframe="1D"
        )
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(history.timestamp, unit='s'),
            'equity': history.equity,
            'profit_loss': history.profit_loss,
            'profit_loss_pct': history.profit_loss_pct
        })
        return df
    except Exception as e:
        st.warning(f"Could not fetch portfolio history: {e}")
        return pd.DataFrame()

def get_recent_orders(api, limit=20):
    """Fetch recent orders."""
    try:
        orders = api.list_orders(status='all', limit=limit)
        
        if not orders:
            return pd.DataFrame()
        
        data = []
        for order in orders:
            data.append({
                'Time': order.submitted_at,
                'Symbol': order.symbol,
                'Side': order.side,
                'Qty': order.qty,
                'Type': order.type,
                'Status': order.status,
                'Filled Price': order.filled_avg_price
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not fetch orders: {e}")
        return pd.DataFrame()

def is_market_open(api):
    """Check if market is open."""
    clock = api.get_clock()
    return clock.is_open, clock.next_open, clock.next_close

# =============================================================================
# DASHBOARD
# =============================================================================

def main():
    # Header
    st.title("RL Portfolio Manager Dashboard")
    st.markdown("**Paper Trading Performance Monitor**")
    
    # Get API client
    api = get_alpaca_client()
    
    # Market status
    is_open, next_open, next_close = is_market_open(api)
    
    if is_open:
        st.success("[OPEN] Market is OPEN")
    else:
        st.warning(f"[CLOSED] Market is CLOSED (Opens: {next_open})")
    
    # Refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # ==========================================================================
    # ACCOUNT OVERVIEW
    # ==========================================================================
    
    st.header("Account Overview")
    
    account = get_account_info(api)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${account['portfolio_value']:,.2f}",
            f"${account['daily_pnl']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Cash Available",
            f"${account['cash']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Buying Power",
            f"${account['buying_power']:,.2f}"
        )
    
    with col4:
        pnl_pct = (account['daily_pnl'] / account['last_equity']) * 100 if account['last_equity'] > 0 else 0
        st.metric(
            "Today's P&L %",
            f"{pnl_pct:+.2f}%"
        )
    
    st.markdown("---")
    
    # ==========================================================================
    # PORTFOLIO CHART
    # ==========================================================================
    
    st.header("Portfolio Performance")
    
    history_df = get_portfolio_history(api, days=30)
    
    if not history_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Add $100k baseline
        fig.add_hline(y=100000, line_dash="dash", line_color="gray",
                      annotation_text="Initial $100k")
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No portfolio history available yet. Start trading to see your performance!")
    
    st.markdown("---")
    
    # ==========================================================================
    # CURRENT POSITIONS
    # ==========================================================================
    
    st.header("Current Positions")
    
    positions_df = get_positions(api)
    
    if not positions_df.empty:
        # Color P&L column
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        st.dataframe(
            positions_df.style.applymap(color_pnl, subset=['P&L', 'P&L %']),
            use_container_width=True
        )
        
        # Position allocation pie chart
        fig = px.pie(
            positions_df,
            values='Market Value',
            names='Symbol',
            title='Portfolio Allocation'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No open positions. The agent hasn't made any trades yet.")
    
    st.markdown("---")
    
    # ==========================================================================
    # RECENT TRADES
    # ==========================================================================
    
    st.header("Recent Orders")
    
    orders_df = get_recent_orders(api, limit=20)
    
    if not orders_df.empty:
        st.dataframe(orders_df, use_container_width=True)
    else:
        st.info("No orders placed yet.")
    
    st.markdown("---")
    
    # ==========================================================================
    # TRADING CONTROLS
    # ==========================================================================
    
    st.header("Trading Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Run Single Trade**")
        st.code("python paper_trading.py --mode single", language="bash")
    
    with col2:
        st.markdown("**Continuous Trading**")
        st.code("python paper_trading.py --mode continuous", language="bash")
    
    with col3:
        st.markdown("**Check Status**")
        st.code("python paper_trading.py --mode status", language="bash")
    
    st.markdown("---")
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        RL Portfolio Manager | Paper Trading Mode | 
        Built with Stable-Baselines3 + FinRL + Alpaca
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()