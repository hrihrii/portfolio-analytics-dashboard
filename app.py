import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import skew, kurtosis
import streamlit as st
from typing import List, Dict, Tuple
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder




# Set Streamlit page configuration
st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

############################
#  Sidebar Configuration   #
############################

with st.sidebar:
    st.markdown("---")


    st.subheader("Benchmark Configuration")
    # New radio to choose the benchmark data source
    use_global_benchmark = st.checkbox("Use Global Benchmark Instead of Per-Asset Benchmarks", value=False)
    benchmark_source = st.radio(
    "Select Benchmark Source",
    options=["Fetch from Ticker", "Upload Benchmark CSV"],
    index=0,
    key="benchmark_source_radio",
    help="Either fetch benchmark data using a ticker or upload a CSV file (with columns: time, price, dividend, return)."
    )

    if benchmark_source == "Fetch from Ticker":
        benchmark_ticker = st.text_input(
            "Benchmark Ticker (e.g., ^GSPC for S&P 500)",
            "^GSPC",
            help="Enter the ticker symbol for the benchmark."
        )
        benchmark_file = None
    else:
        benchmark_ticker = ""
        benchmark_file = st.file_uploader(
            "Upload Benchmark CSV",
            type=["csv"],
            key="benchmark_file",
            help="Upload a CSV file with columns: time, price, dividend, return."
        )
    st.markdown("---")
    
    # **NEW: Risk-Free Rate Uploader**
    st.subheader("Risk‑Free Rate Data")
    risk_free_file = st.file_uploader(
        "Upload Risk‑Free Rate CSV",
        type=["csv"],
        key="risk_free_file",
        help="Upload a CSV file with columns: date (or time) and risk_free_rate."
    )
    st.markdown("---")

############################
#  Process Risk‑Free Rate  #
############################

risk_free_df = pd.DataFrame()
if risk_free_file is not None:
    risk_free_df = pd.read_csv(risk_free_file)
    risk_free_df.columns = risk_free_df.columns.str.strip().str.lower()
    if 'date' in risk_free_df.columns:
        risk_free_df = risk_free_df.rename(columns={'date': 'time'})
    risk_free_df['time'] = pd.to_datetime(risk_free_df['time'])
    if 'risk_free_rate' in risk_free_df.columns:
        risk_free_df['risk_free_rate'] = risk_free_df['risk_free_rate'].astype(float) / 100
        risk_free_df['risk_free_rate'] = risk_free_df['risk_free_rate'].astype(float)

        # Standardize to end-of-month timestamps (if desired)
        risk_free_df['time'] = risk_free_df['time'].dt.to_period('M').dt.to_timestamp('M')


############################
#  Helper Functions        #
############################





def align_time_indices(portfolio_returns: pd.Series, benchmark_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])
    benchmark_data.set_index('time', inplace=True)
    common_dates = portfolio_returns.index.intersection(benchmark_data.index)
    aligned_portfolio_returns = portfolio_returns.loc[common_dates]
    aligned_benchmark_returns = benchmark_data.loc[common_dates, 'return']
    st.write("DEBUG: Aligned Portfolio Returns")
    st.dataframe(aligned_portfolio_returns.head())
    st.write("DEBUG: Aligned Benchmark Returns")
    st.dataframe(aligned_benchmark_returns.head())
    return aligned_portfolio_returns, aligned_benchmark_returns

def validate_time_ranges(asset_data: List[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_dates = [df['time'].min() for df in asset_data if not df.empty]
    max_dates = [df['time'].max() for df in asset_data if not df.empty]
    if not min_dates or not max_dates:
        raise ValueError("No valid time ranges found in asset data.")
    start_date = max(min_dates)  # Latest start date
    end_date = min(max_dates)    # Earliest end date
    if start_date > end_date:
        raise ValueError("Assets do not have overlapping time ranges.")
    return start_date, end_date

def normalize_weights(weights: List[float]) -> List[float]:
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    return [w / total_weight for w in weights]

def plot_growth_of_assets(asset_data: List[pd.DataFrame], asset_names: List[str], portfolio_returns: pd.Series, benchmark_data: pd.DataFrame) -> None:
    fig = go.Figure()
    chart_dfs = []
    for i, df in enumerate(asset_data):
        if df.empty:
            st.warning(f"Asset data for {asset_names[i]} is empty. Skipping this asset.")
            continue
        tmp = df[['time']].copy()
        tmp['Growth'] = (1 + df['total_return']).cumprod()
        tmp['Label'] = asset_names[i]
        chart_dfs.append(tmp)
        fig.add_trace(go.Scatter(
            x=tmp['time'],
            y=tmp['Growth'],
            mode='lines',
            name=asset_names[i],
            hovertemplate="Time: %{x}<br>Growth: %{y:.2f}<extra></extra>"
        ))
    if not portfolio_returns.empty:
        port_tmp = pd.DataFrame({'time': portfolio_returns.index})
        port_tmp['Growth'] = (1 + portfolio_returns).cumprod()
        port_tmp['Label'] = 'Portfolio'
        chart_dfs.append(port_tmp)
        fig.add_trace(go.Scatter(
            x=port_tmp['time'],
            y=port_tmp['Growth'],
            mode='lines',
            name='Portfolio',
            line=dict(color='black', dash='dash'),
            hovertemplate="Time: %{x}<br>Growth: %{y:.2f}<extra></extra>"
        ))
    else:
        st.warning("Portfolio returns are empty. Portfolio line will not be displayed.")
    if not benchmark_data.empty:
        if 'time' in benchmark_data.columns and 'return' in benchmark_data.columns:
            benchmark_tmp = benchmark_data[['time']].copy()
            benchmark_tmp['Growth'] = (1 + benchmark_data['return']).cumprod()
            benchmark_tmp['Label'] = 'Benchmark'
            chart_dfs.append(benchmark_tmp)
            fig.add_trace(go.Scatter(
                x=benchmark_tmp['time'],
                y=benchmark_tmp['Growth'],
                mode='lines',
                name='Benchmark',
                line=dict(color='red'),
                hovertemplate="Time: %{x}<br>Growth: %{y:.2f}<extra></extra>"
            ))
        else:
            st.warning("Benchmark data is missing required columns ('time', 'return').")
    if chart_dfs:
        full_data = pd.concat(chart_dfs, ignore_index=True)
        full_data['time'] = pd.to_datetime(full_data['time'])
        st.write("DEBUG: DataFrame being plotted for portfolio growth:")
        st.dataframe(full_data)
    else:
        st.error("No data available to plot. Check inputs.")
        return
    fig.update_layout(
        title="Cumulative Growth of Assets and Portfolio",
        xaxis_title="Time",
        yaxis_title="Cumulative Growth",
        template="plotly_white",
        legend_title="Legend",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_drawdown_of_assets(asset_data: List[pd.DataFrame], asset_names: List[str], portfolio_returns: pd.Series) -> None:
    chart_dfs = []
    common_index = portfolio_returns.index  # Use portfolio_returns as the reference index

    # Handle single asset case
    if len(asset_data) == 1:
        single_asset = asset_data[0]
        if single_asset.empty:
            st.warning("Asset data is empty. Drawdown chart cannot be displayed.")
            return
        # Reindex to common_index
        total_returns = single_asset.set_index('time')['total_return'].reindex(common_index, method='ffill')
        cumret = (1 + total_returns).cumprod()
        runmax = cumret.cummax()
        drawdown = (cumret / runmax) - 1
        tmp = pd.DataFrame({
            'time': common_index,
            'Drawdown': drawdown,
            'Label': asset_names[0]
        })
        chart_dfs.append(tmp)
        full_data = pd.concat(chart_dfs, ignore_index=True)
        full_data['time'] = pd.to_datetime(full_data['time'])
        if full_data.empty or full_data['Drawdown'].isna().all():
            st.error("No valid drawdown data for the single asset. Check input data.")
            return
        single_drawdown_chart = alt.Chart(full_data).mark_line().encode(
            x=alt.X('time:T', title='Time'),
            y=alt.Y('Drawdown:Q', title='Drawdown', scale=alt.Scale(domain=[-1.0, 0])),
            color=alt.Color('Label:N', legend=alt.Legend(title="Legend")),
            tooltip=[alt.Tooltip('time:T', title='Date', format='%Y-%m-%d'),
                     alt.Tooltip('Drawdown:Q', title='Drawdown', format='.2%')]
        ).interactive()
        st.altair_chart(single_drawdown_chart, use_container_width=True)
        return

    # Multiple assets case
    for i, df in enumerate(asset_data):
        if df.empty:
            st.warning(f"Asset data for {asset_names[i]} is empty. Skipping in drawdown plot.")
            continue
        # Reindex to common_index to ensure alignment
        total_returns = df.set_index('time')['total_return'].reindex(common_index, method='ffill')
        if total_returns.isna().all():
            st.warning(f"No valid total_return data for {asset_names[i]} after alignment. Skipping.")
            continue
        cumret = (1 + total_returns).cumprod()
        runmax = cumret.cummax()
        drawdown = (cumret / runmax) - 1
        tmp = pd.DataFrame({
            'time': common_index,
            'Drawdown': drawdown,
            'Label': asset_names[i]
        })
        chart_dfs.append(tmp)
        

    # Portfolio drawdown
    port_cumret = (1 + portfolio_returns).cumprod()
    port_runmax = port_cumret.cummax()
    port_drawdown = (port_cumret / port_runmax) - 1
    port_df = pd.DataFrame({
        'time': common_index,
        'Drawdown': port_drawdown.values,
        'Label': 'Portfolio'
    })
    chart_dfs.append(port_df)
    

    # Concatenate and plot
    if not chart_dfs:
        st.error("No valid drawdown data to plot. Check asset_data inputs.")
        return
    full_data = pd.concat(chart_dfs, ignore_index=True)
    full_data['time'] = pd.to_datetime(full_data['time'])
    

    if full_data.empty or full_data['Drawdown'].isna().all():
        st.error("No valid drawdown data after concatenation. Check inputs.")
        return

    selection = alt.selection_multi(fields=['Label'], bind='legend')
    color_scale = alt.Scale(
        domain=asset_names + ['Portfolio'],
        range=[
                'orange', 'green', 'purple', 'brown', 'steelblue', 'hotpink', 'gray', 'teal', 'indigo',
                'red', 'cyan', 'magenta', 'yellow', 'lime', 'gold', 'maroon', 'navy', 'violet', 'olive',
                'coral', 'turquoise', 'salmon', 'darkgreen', 'peru', 'orchid', 'crimson', 'sienna', 'plum',
                'khaki', 'tan', 'lavender', 'chartreuse', 'firebrick', 'tomato', 'slategray', 'darkorange'
            ][:len(asset_names)] + ['blue']  # Dark blue for Portfolio
    )
    drawdown_chart = alt.Chart(full_data).mark_line().encode(
        x=alt.X('time:T', title='Time'),
        y=alt.Y('Drawdown:Q', title='Drawdown', scale=alt.Scale(domain=[-1.0, 0])),
        color=alt.Color('Label:N', scale=color_scale),
        tooltip=[alt.Tooltip('time:T', title='Date', format='%Y-%m-%d'),
                 alt.Tooltip('Label:N', title='Asset'),
                 alt.Tooltip('Drawdown:Q', title='Drawdown', format='.2%')]
    ).add_selection(
        selection
    ).transform_filter(
        selection
    ).interactive()
    st.altair_chart(drawdown_chart, use_container_width=True)
    
def fetch_benchmark_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        benchmark_data = yf.download(
            ticker, start=start_date, end=end_date, interval="1mo"
        )
        benchmark_data.reset_index(inplace=True)
        mapped_data = pd.DataFrame()
        if 'Date' in benchmark_data.columns:
            mapped_data['time'] = pd.to_datetime(benchmark_data['Date'])
            mapped_data['time'] = mapped_data['time'].dt.to_period('M').dt.to_timestamp('M')
        else:
            raise ValueError("The 'Date' column is missing from the benchmark data.")
        if 'Adj Close' in benchmark_data.columns:
            mapped_data['price'] = benchmark_data['Adj Close']
        elif 'Close' in benchmark_data.columns:
            mapped_data['price'] = benchmark_data['Close']
        else:
            raise ValueError("No valid price column ('Adj Close' or 'Close') found in the benchmark data.")
        mapped_data['return'] = mapped_data['price'].pct_change()
        mapped_data.dropna(subset=['return'], inplace=True)
        return mapped_data[['time', 'price', 'return']]
    except Exception as e:
        raise ValueError(f"Error processing benchmark data: {e}")

def adjust_column_names(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = ['time', 'price', 'dividend', 'return']
    df.columns = df.columns.str.strip().str.lower()
    existing_cols = df.columns.tolist()
    rename_map = {}
    for i, expected_col in enumerate(expected_columns):
        if i < len(existing_cols):
            rename_map[existing_cols[i]] = expected_col
    return df.rename(columns=rename_map)

def validate_columns(df: pd.DataFrame) -> None:
    # **UPDATED: Remove 'risk_free_rate' requirement**
    required_columns = {'time', 'price', 'dividend', 'return'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def calculate_metrics(df: pd.DataFrame, dividend_as_yield: bool = True) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    if df.empty:
        raise ValueError("Empty DataFrame provided.")
    df = df.sort_values('time').copy()
    df['dividend'] = df['dividend'].fillna(0)
    if dividend_as_yield:
        df['total_return'] = df['price'].pct_change() + (df['dividend'] / 100)
    else:
        df['total_return'] = (df['price'] + df['dividend']) / df['price'].shift(1) - 1
    df.dropna(subset=['total_return'], inplace=True)  # Only drop NaN for total_return
    if len(df) < 2:
        raise ValueError("Insufficient data points for analysis.")
    if 'risk_free_rate' in df.columns:
        df['excess_return'] = df['total_return'] - df['risk_free_rate'].fillna(0)
    else:
        df['excess_return'] = df['total_return']
        
    df['excess_return'] = df['total_return'] - df['risk_free_rate']
    raw_volatility = df['total_return'].std()
    raw_mean_return = df['total_return'].mean()
    raw_geo_mean_return = (1 + df['total_return']).prod() ** (1 / len(df)) - 1
    cumulative_returns = (1 + df['total_return']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown_series = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown_series.min()
    raw_metrics = {
        'Volatility (Monthly)': raw_volatility,
        'Arithmetic Mean Return (Monthly)': raw_mean_return,
        'Geometric Mean Return (Monthly)': raw_geo_mean_return,
        'Maximum Drawdown': max_drawdown,
        'Sharpe Ratio (Monthly)': df['excess_return'].mean() / df['excess_return'].std() if df['excess_return'].std() > 0 else 0,
        'Skewness': skew(df['total_return']),
        'Value at Risk (95%)': np.percentile(df['total_return'], 5),
        'Conditional Value at Risk (95%)': df['total_return'][df['total_return'] <= np.percentile(df['total_return'], 5)].mean()
    }
    annual_volatility = raw_volatility * np.sqrt(12)
    annual_arithmetic_mean = raw_mean_return * 12
    annual_geometric_mean = (1 + raw_geo_mean_return) ** 12 - 1
    annual_metrics = {
        'Volatility (Annual)': annual_volatility,
        'Arithmetic Mean Return (Annual)': annual_arithmetic_mean,
        'Geometric Mean Return (Annual)': annual_geometric_mean,
        'Maximum Drawdown': max_drawdown,
        'Sharpe Ratio (Annual)': raw_metrics['Sharpe Ratio (Monthly)'] * np.sqrt(12)
    }
    return df, raw_metrics, annual_metrics

def align_asset_data(asset_data: List[pd.DataFrame], asset_names: List[str], benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
    named_dfs = []
    for i, df in enumerate(asset_data):
        if df.empty:
            st.warning(f"Asset data for {asset_names[i]} is empty. Skipping this asset.")
            continue
        df = df.sort_values('time').copy()
        df.rename(columns={'time': 'time_col'}, inplace=True)
        df.set_index('time_col', inplace=True)
        named_dfs.append(df[['total_return']].rename(columns={'total_return': asset_names[i]}))
    if not named_dfs:
        st.error("No asset data available for alignment.")
        return pd.DataFrame()
    aligned = pd.concat(named_dfs, axis=1, join='inner')
    aligned = aligned[~aligned.index.duplicated(keep='first')]
    aligned.dropna(how='all', inplace=True)
    return aligned

def display_metrics(raw_metrics: Dict[str, float], annualized_metrics: Dict[str, float]) -> None:
    percentage_keys = {
        'Volatility (Monthly)', 'Volatility (Annual)',
        'Arithmetic Mean Return (Monthly)', 'Arithmetic Mean Return (Annual)',
        'Maximum Drawdown',
        'Upside Capture Ratio', 'Downside Capture Ratio',
        'Calmar Ratio', 'Treynor Ratio',
        'Tracking Error',
        'Value at Risk (95%)', 'Conditional Value at Risk (95%)',
        'Alpha'
    }
    def format_metric_value(key: str, val: float) -> str:
        if not isinstance(val, (int, float)):
            return str(val)
        if key in percentage_keys:
            return f"{val:.2%}"
        else:
            return f"{val:.4f}"
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Metrics (Monthly)")
        raw_metrics_df = pd.DataFrame({
            'Metric': list(raw_metrics.keys()),
            'Value': [format_metric_value(k, v) for k, v in raw_metrics.items()]
        })
        st.dataframe(raw_metrics_df)
    with col2:
        st.subheader("Annualized Metrics")
        annual_metrics_df = pd.DataFrame({
            'Metric': list(annualized_metrics.keys()),
            'Value': [format_metric_value(k, v) for k, v in annualized_metrics.items()]
        })
        st.dataframe(annual_metrics_df)

def compute_risk_contributions(aligned_returns: pd.DataFrame, weights: List[float]) -> pd.DataFrame:
    if aligned_returns.empty:
        raise ValueError("Aligned returns DataFrame is empty. Cannot compute risk contributions.")
    if aligned_returns.shape[1] != len(weights):
        raise ValueError("Mismatch in number of columns vs. number of weights.")
    cov_matrix = aligned_returns.cov()
    if cov_matrix.isnull().values.any():
        raise ValueError("Covariance matrix contains NaN values. Check your input data.")
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    if port_vol == 0:
        raise ValueError("Portfolio volatility is zero. Check your input data or weights.")
    mctr_list = []
    for i in range(len(weights)):
        cov_i_p = np.dot(cov_matrix.iloc[i, :], weights)
        mctr = weights[i] * cov_i_p / port_vol
        mctr_list.append(mctr)
    pct_list = [(m / port_vol) for m in mctr_list]
    df_risk = pd.DataFrame({
        'Weight': weights,
        'MCTR': mctr_list,
        'Percent of Risk': [p * 100 for p in pct_list]
    }, index=aligned_returns.columns)
    return df_risk
def calc_periodic_metrics(fund_returns: pd.Series, bench_returns: pd.Series, risk_free: pd.Series) -> Dict[str, float]:
    # Ensure the series are sorted by date.
    common_idx = fund_returns.index.intersection(bench_returns.index).intersection(risk_free.index)
    fund_returns = fund_returns.sort_index()
    bench_returns = bench_returns.sort_index()
    risk_free = risk_free.sort_index()
    N = len(fund_returns)
    if N == 0:
        return {metric: None for metric in [
            "Ann Return (Fund)", "Ann Return (Benchmark)", "Excess Return", "Ann Risk-free Rate",
            "Ann Std Dev (Fund)", "Ann Std Dev (Benchmark)", "Downside Deviation (Fund)", "Downside Deviation (Benchmark)",
            "Maximum Drawdown (Fund)", "Maximum Drawdown (Benchmark)", "Win/Loss Ratio (Fund)", "Win/Loss Ratio (Benchmark)",
            "Up Capture", "Down Capture", "Sharpe Ratio (Fund)", "Sharpe Ratio (Benchmark)",
            "Beta", "Alpha", "Tracking Error (Fund)", "Information Ratio"
        ]}
    # Annualized returns:
    ann_return_fund = (1 + fund_returns).prod()**(12 / N) - 1
    ann_return_bench = (1 + bench_returns).prod()**(12 / N) - 1
    excess_return = ann_return_fund - ann_return_bench
    # Annual risk-free rate (assumed constant over period)
    ann_risk_free = (1 + risk_free.mean()) ** 12 - 1
    # Annualized standard deviations:
    ann_std_fund = fund_returns.std() * np.sqrt(12)
    ann_std_bench = bench_returns.std() * np.sqrt(12)
    # Downside deviation:
    downside_fund = fund_returns[fund_returns < 0].std() * np.sqrt(12) if not fund_returns[fund_returns < 0].empty else np.nan
    downside_bench = bench_returns[bench_returns < 0].std() * np.sqrt(12) if not bench_returns[bench_returns < 0].empty else np.nan
    # Maximum drawdown:
    cum_fund = (1 + fund_returns).cumprod()
    max_dd_fund = (cum_fund / cum_fund.cummax() - 1).min()
    cum_bench = (1 + bench_returns).cumprod()
    max_dd_bench = (cum_bench / cum_bench.cummax() - 1).min()
    # Win/Loss Ratio:
    win_loss_fund = (fund_returns[fund_returns > 0].mean() / abs(fund_returns[fund_returns < 0].mean())) if (not fund_returns[fund_returns < 0].empty and fund_returns[fund_returns < 0].mean() != 0) else np.nan
    win_loss_bench = (bench_returns[bench_returns > 0].mean() / abs(bench_returns[bench_returns < 0].mean())) if (not bench_returns[bench_returns < 0].empty and bench_returns[bench_returns < 0].mean() != 0) else np.nan
    # Up Capture:
    up_bench = bench_returns[bench_returns > 0]
    up_fund = fund_returns[fund_returns.index.isin(up_bench.index)]
    if not up_bench.empty:
        geo_up_bench = np.exp(np.log1p(up_bench).mean()) - 1
        geo_up_fund  = np.exp(np.log1p(up_fund).mean()) - 1
        up_capture = (geo_up_fund / geo_up_bench) if geo_up_bench != 0 else np.nan
    else:
        up_capture = np.nan
    # Down Capture:
    down_bench = bench_returns[bench_returns < 0]
    down_fund = fund_returns[fund_returns.index.isin(down_bench.index)]
    if not down_bench.empty:
        geo_down_bench = np.exp(np.log1p(down_bench).mean()) - 1
        geo_down_fund  = np.exp(np.log1p(down_fund).mean()) - 1
        down_capture = (geo_down_fund / geo_down_bench) if geo_down_bench != 0 else np.nan
    else:
        down_capture = np.nan
    # Sharpe Ratios:
    sharpe_fund = (ann_return_fund - ann_risk_free) / ann_std_fund if ann_std_fund > 0 else np.nan
    sharpe_bench = (ann_return_bench - ann_risk_free) / ann_std_bench if ann_std_bench > 0 else np.nan
    # Beta and Alpha:
    cov = np.cov(fund_returns, bench_returns)[0, 1]
    var_bench = bench_returns.var()
    beta = cov / var_bench if var_bench > 0 else np.nan
    alpha = ann_return_fund - (ann_risk_free + beta * (ann_return_bench - ann_risk_free))
    # Tracking Error and Information Ratio:
    tracking_error_fund = (fund_returns - bench_returns).std() * np.sqrt(12)
    info_ratio = excess_return / tracking_error_fund if tracking_error_fund > 0 else np.nan

    return {
        "Ann Return (Fund)": ann_return_fund,
        "Ann Return (Benchmark)": ann_return_bench,
        "Excess Return": excess_return,
        "Ann Risk-free Rate": ann_risk_free,
        "Ann Std Dev (Fund)": ann_std_fund,
        "Ann Std Dev (Benchmark)": ann_std_bench,
        "Downside Deviation (Fund)": downside_fund,
        "Downside Deviation (Benchmark)": downside_bench,
        "Maximum Drawdown (Fund)": max_dd_fund,
        "Maximum Drawdown (Benchmark)": max_dd_bench,
        "Win/Loss Ratio (Fund)": win_loss_fund,
        "Win/Loss Ratio (Benchmark)": win_loss_bench,
        "Up Capture": up_capture,
        "Down Capture": down_capture,
        "Sharpe Ratio (Fund)": sharpe_fund,
        "Sharpe Ratio (Benchmark)": sharpe_bench,
        "Beta": beta,
        "Alpha": alpha,
        "Tracking Error (Fund)": tracking_error_fund,
        "Information Ratio": info_ratio
    }

# **UPDATED: Modify portfolio_metrics to accept risk_free_df**
def portfolio_metrics(asset_data: List[pd.DataFrame], weights: List[float], asset_names: List[str], 
                     benchmark_data: pd.DataFrame = None, risk_free_df: pd.DataFrame = None, 
                     asset_benchmarks: List[pd.DataFrame] = None) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], pd.DataFrame, pd.Series]:
    if not asset_data or not weights:
        raise ValueError("Asset data and weights cannot be empty.")
    if len(asset_data) != len(weights):
        raise ValueError("Number of assets must match number of weights.")
    if asset_benchmarks and len(asset_data) != len(asset_benchmarks):
        raise ValueError("Number of asset benchmarks must match number of assets.")
    
    # Risk-free rate handling
    if risk_free_df is not None and not risk_free_df.empty:
        risk_free_rate = risk_free_df.set_index('time')['risk_free_rate']
    else:
        risk_free_rate = pd.Series(0.0, index=align_asset_data(asset_data, asset_names).index)

    # Align asset returns
    aligned_returns = align_asset_data(asset_data, asset_names)
    portfolio_returns = aligned_returns.dot(weights)
    excess_returns = portfolio_returns - risk_free_rate.reindex(portfolio_returns.index).fillna(0)

    # Compute weighted benchmark returns if per-asset benchmarks are provided
    if asset_benchmarks and all(not b.empty for b in asset_benchmarks):
        weighted_bench_df = pd.DataFrame(index=aligned_returns.index)
        for i, bench_df in enumerate(asset_benchmarks):
            if not bench_df.empty:
                bench_returns = bench_df.set_index('time')['return'].reindex(aligned_returns.index).fillna(0)
                weighted_bench_df[asset_names[i]] = bench_returns * weights[i]
            else:
                st.warning(f"No benchmark data for {asset_names[i]}. Using zero returns.")
                weighted_bench_df[asset_names[i]] = pd.Series(0.0, index=aligned_returns.index)
        weighted_benchmark_returns = weighted_bench_df.sum(axis=1)
    elif benchmark_data is not None and not benchmark_data.empty:
        weighted_benchmark_returns = benchmark_data.set_index('time')['return'].reindex(aligned_returns.index).fillna(0)
    else:
        weighted_benchmark_returns = pd.Series(0.0, index=aligned_returns.index)

    # Align all series to a common index once
    common_index = aligned_returns.index.intersection(weighted_benchmark_returns.index).intersection(risk_free_rate.index)
    if common_index.empty:
        raise ValueError("No common dates found across portfolio returns, benchmark returns, and risk-free rate.")
    
    portfolio_returns = portfolio_returns.reindex(common_index)
    excess_returns = excess_returns.reindex(common_index)
    weighted_benchmark_returns = weighted_benchmark_returns.reindex(common_index)
    risk_free_rate = risk_free_rate.reindex(common_index).fillna(0)

    

    # Compute risk decomposition and metrics
    risk_decomp_table = compute_risk_contributions(aligned_returns.loc[common_index], weights)
    
    # Call _calculate_metrics and unpack its results, then add weighted_benchmark_returns to the return tuple
    portfolio_returns, raw_metrics, annual_metrics, risk_decomp_table = _calculate_metrics(
        portfolio_returns, excess_returns, risk_free_rate, weighted_benchmark_returns, risk_decomp_table
    )
    return portfolio_returns, raw_metrics, annual_metrics, risk_decomp_table, weighted_benchmark_returns

def _calculate_metrics(portfolio_returns, excess_returns, risk_free_rate, benchmark_returns, risk_decomp_table: pd.DataFrame = None) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], pd.DataFrame]:
    # Remove redundant alignment since it's handled in portfolio_metrics
    volatility_monthly = portfolio_returns.std()
    mean_return_monthly = portfolio_returns.mean()
    sharpe_monthly = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns) ** 2))
    sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown_series = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown_series.min()
    volatility_annual = volatility_monthly * np.sqrt(12)
    mean_return_annual = (1 + mean_return_monthly) ** 12 - 1
    sharpe_annual = sharpe_monthly * np.sqrt(12)
    calmar_ratio = mean_return_annual / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Debug before covariance calculation
    
    
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    beta = covariance / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
    risk_free_rate_annual = (1 + risk_free_rate.mean()) ** 12 - 1
    mean_benchmark_return_monthly = benchmark_returns.mean()
    mean_benchmark_return_annual = mean_benchmark_return_monthly * 12
    alpha = mean_return_annual - (risk_free_rate_annual + beta * (mean_benchmark_return_annual - risk_free_rate_annual))
    tracking_error = (portfolio_returns - benchmark_returns).std()
    information_ratio = (mean_return_monthly - mean_benchmark_return_monthly) / tracking_error if tracking_error > 0 else 0
    
    benchmark_up = benchmark_returns[benchmark_returns > 0]
    portfolio_up = portfolio_returns.loc[benchmark_up.index.intersection(portfolio_returns.index)]
    benchmark_down = benchmark_returns[benchmark_returns < 0]
    portfolio_down = portfolio_returns.loc[benchmark_down.index.intersection(portfolio_returns.index)]
    
    if not benchmark_up.empty:
        upside_capture = (portfolio_up.mean() / benchmark_up.mean() if benchmark_up.mean() > 0 else 0)
    else:
        upside_capture = 0
    if not benchmark_down.empty:
        downside_capture = (portfolio_down.mean() / benchmark_down.mean() if benchmark_down.mean() < 0 else 0)
    else:
        downside_capture = 0
    
    raw_metrics = {
        'Volatility (Monthly)': volatility_monthly,
        'Arithmetic Mean Return (Monthly)': mean_return_monthly,
        'Sharpe Ratio (Monthly)': sharpe_monthly,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Tracking Error': tracking_error,
        'Information Ratio': information_ratio,
        'Upside Capture Ratio': upside_capture,
        'Downside Capture Ratio': downside_capture,
    }
    annual_metrics = {
        'Volatility (Annual)': volatility_annual,
        'Arithmetic Mean Return (Annual)': mean_return_annual,
        'Sharpe Ratio (Annual)': sharpe_annual,
        'Calmar Ratio': calmar_ratio,
        'Alpha': alpha,
        'Beta': beta,
    }
    
    # Return the 4-tuple expected by portfolio_metrics, which will add the fifth element
    if risk_decomp_table is not None:
        return portfolio_returns, raw_metrics, annual_metrics, risk_decomp_table
    else:
        return portfolio_returns, raw_metrics, annual_metrics, pd.DataFrame()
    
def calc_annualized_slice_metrics(returns_slice: pd.Series) -> Dict[str, float]:
    returns_slice = returns_slice.dropna()
    if returns_slice.empty:
        return {'Arithmetic': None, 'Geometric': None, 'Volatility': None, 'MaxDD': None, 'Sharpe': None}
    monthly_mean = returns_slice.mean()
    monthly_std  = returns_slice.std()
    n_months     = len(returns_slice)
    total_growth = (1 + returns_slice).prod()
    annual_arithmetic = monthly_mean * 12
    geo = total_growth ** (12 / n_months) - 1 if n_months >= 1 else None
    annual_vol = monthly_std * (12 ** 0.5) if monthly_std is not None else None
    cum = (1 + returns_slice).cumprod()
    running_max = cum.cummax()
    dd_series = (cum / running_max) - 1
    max_dd = dd_series.min()
    if monthly_std > 0:
        # Calculate the monthly geometric return.
        monthly_geom = total_growth ** (1 / n_months) - 1
        monthly_sharpe = monthly_geom / monthly_std
        annual_sharpe  = monthly_sharpe * (12 ** 0.5)
    else:
        annual_sharpe  = None

    return {
        'Arithmetic': annual_arithmetic,
        'Geometric':  geo,
        'Volatility': annual_vol,
        'MaxDD':      max_dd,
        'Sharpe':     annual_sharpe
    }
def display_individual_asset_periodic_metrics(asset_returns_dict: Dict[str, pd.Series],
                                              asset_benchmarks_dict: Dict[str, pd.Series],
                                              risk_free_series: pd.Series,
                                              periods: List[str] = ["1Y", "3Y", "5Y", "ITD"]) -> None:
    groups = {}
    common_index = risk_free_series.index
    for asset, returns_series in asset_returns_dict.items():
        returns_series = returns_series.reindex(common_index, method='ffill')
        bench_returns = asset_benchmarks_dict.get(asset, pd.Series(0.0, index=common_index))
        bench_returns = bench_returns.reindex(common_index, method='ffill')
        if bench_returns.empty or bench_returns.isna().all():
            bench_returns = pd.Series(0.0, index=common_index)

        asset_metrics = {}
        for period in periods:
            if period == "ITD":
                # Use full series, already aligned to common_index
                sliced_returns = returns_series
                sliced_bench = bench_returns
                sliced_risk_free = risk_free_series
            else:
                months = int(period[:-1]) * 12
                end_date = returns_series.index.max()
                start_date = end_date - pd.DateOffset(months=months)
                # Define a period-specific index
                period_index = returns_series.index[(returns_series.index >= start_date) & (returns_series.index <= end_date)]
                if period_index.empty:
                    st.warning(f"No data available for {asset} in period {period}. Skipping.")
                    asset_metrics[period] = {metric: None for metric in calc_periodic_metrics(pd.Series(), pd.Series(), pd.Series())}
                    continue
                # Reindex all series to the period-specific index
                sliced_returns = returns_series.reindex(period_index, method='ffill')
                sliced_bench = bench_returns.reindex(period_index, method='ffill')
                sliced_risk_free = risk_free_series.reindex(period_index, method='ffill').fillna(0.0)

            # Ensure lengths match
            
            metrics = calc_periodic_metrics(sliced_returns, sliced_bench, sliced_risk_free)
            asset_metrics[period] = metrics
        groups[asset] = asset_metrics

    # Build a DataFrame with MultiIndex columns (Asset, Period) and rows = metric names.
    metric_names = list(next(iter(groups.values()))["ITD"].keys())
    col_tuples = []
    data_dict = {metric: [] for metric in metric_names}
    for asset, period_metrics in groups.items():
        for period in periods:
            col_tuples.append((asset, period))
            for metric in metric_names:
                data_dict[metric].append(period_metrics[period][metric])
    col_index = pd.MultiIndex.from_tuples(col_tuples, names=["Asset", "Period"])
    df_extended = pd.DataFrame(data_dict, index=col_index).T

    # Define which metrics should be formatted as percentages and which use a specific format.
    percentage_metrics = {
        "Ann Return (Fund)",
        "Ann Return (Benchmark)",
        "Excess Return",
        "Ann Risk-free Rate",
        "Ann Std Dev (Fund)",
        "Ann Std Dev (Benchmark)",
        "Downside Deviation (Fund)",
        "Downside Deviation (Benchmark)",
        "Maximum Drawdown (Fund)",
        "Maximum Drawdown (Benchmark)",
        "Alpha",
        "Tracking Error (Fund)"
    }

    # Define specific formatting rules for non-percentage metrics (except Up/Down Capture)
    format_dict = {
        "Win/Loss Ratio (Fund)": "{:.3f}",
        "Win/Loss Ratio (Benchmark)": "{:.3f}",
        "Sharpe Ratio (Fund)": "{:.3f}",
        "Sharpe Ratio (Benchmark)": "{:.3f}",
        "Beta": "{:.3f}",
        "Information Ratio": "{:.3f}"
    }

    # Use the row's index (i.e. the metric name) to decide formatting for that entire row.
    def format_row(row):
        metric_name = row.name  # e.g., "Up Capture"
        if metric_name in percentage_metrics:
            return row.apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        elif metric_name in {"Up Capture", "Down Capture"}:
            # Multiply by 100 so that 1.04 shows as 104%
            return row.apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        elif metric_name in format_dict:
            return row.apply(lambda x: format_dict[metric_name].format(x) if pd.notna(x) else "N/A")
        else:
            return row.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")



    formatted_df = df_extended.apply(format_row, axis=1)

    st.subheader("Individual Asset Periodic Metrics")
    st.table(formatted_df)
    csv_data = formatted_df.to_csv(index=True)
    st.download_button(
        label="Download Individual Asset Periodic Metrics CSV",
        data=csv_data,
        file_name="individual_asset_periodic_metrics.csv",
        mime="text/csv",
        key=f"download_individual_metrics_{asset}"
        )
    
    table_data = []
    # Iterate over each asset in the asset_returns_dict.
    for asset, returns_series in asset_returns_dict.items():
        if returns_series.empty:
            continue
        # Determine the common end date for the asset.
        end_date = returns_series.index.max()
        # Inception-to-date (ITD) uses all available data.
        itd_start = returns_series.index.min()
        # Fixed windows: 1Y (12 months), 3Y (36 months), and 5Y (60 months).
        one_year_start = end_date - pd.DateOffset(months=12)
        three_year_start = end_date - pd.DateOffset(months=36)
        five_year_start = end_date - pd.DateOffset(months=60)
        
        # Append a row for each period.
        table_data.append({
            "Asset": asset,
            "Period": "1Y",
            "Start Date": one_year_start.strftime("%Y-%m-%d"),
            "End Date": end_date.strftime("%Y-%m-%d")
        })
        table_data.append({
            "Asset": asset,
            "Period": "3Y",
            "Start Date": three_year_start.strftime("%Y-%m-%d"),
            "End Date": end_date.strftime("%Y-%m-%d")
        })
        table_data.append({
            "Asset": asset,
            "Period": "5Y",
            "Start Date": five_year_start.strftime("%Y-%m-%d"),
            "End Date": end_date.strftime("%Y-%m-%d")
        })
        table_data.append({
            "Asset": asset,
            "Period": "ITD",
            "Start Date": itd_start.strftime("%Y-%m-%d"),
            "End Date": end_date.strftime("%Y-%m-%d")
        })
    
    # Create a DataFrame from the table data.
    df_dates = pd.DataFrame(table_data)
    
    # Optionally, sort the table for clarity.
    df_dates = df_dates.sort_values(["Asset", "Period"])
    
    st.subheader("Data Window Details")
    st.table(df_dates)

def display_full_periodic_table(fund_returns: pd.Series, bench_returns: pd.Series, risk_free_series: pd.Series) -> None:
    st.subheader("Periodic Annualized Stats: 1Y, 3Y, 5Y, ITD")

    def slice_and_calc(months: int) -> Dict[str, float]:
        end_date = fund_returns.index.max()
        start_date = end_date - pd.DateOffset(months=months)
        fund_slice = fund_returns.loc[start_date:end_date]
        bench_slice = bench_returns.loc[start_date:end_date]
        risk_slice = risk_free_series.loc[start_date:end_date]
        return calc_periodic_metrics(fund_slice, bench_slice, risk_slice)

    one_year = slice_and_calc(12)
    three_year = slice_and_calc(36)
    five_year = slice_and_calc(60)
    itd = calc_periodic_metrics(fund_returns, bench_returns, risk_free_series)

    # Build the DataFrame
    periods = ["1Y", "3Y", "5Y", "ITD"]
    metrics = list(itd.keys())
    data = {period: [] for period in periods}
    for metric in metrics:
        data["1Y"].append(one_year[metric])
        data["3Y"].append(three_year[metric])
        data["5Y"].append(five_year[metric])
        data["ITD"].append(itd[metric])
    df_table = pd.DataFrame(data, index=metrics)

    # Define which metrics should be formatted as percentages
        # Define which metrics should be formatted as percentages
    percentage_metrics = {
        "Ann Return (Fund)",
        "Ann Return (Benchmark)",
        "Excess Return",
        "Ann Risk-free Rate",
        "Ann Std Dev (Fund)",
        "Ann Std Dev (Benchmark)",
        "Downside Deviation (Fund)",
        "Downside Deviation (Benchmark)",
        "Maximum Drawdown (Fund)",
        "Maximum Drawdown (Benchmark)",
        "Alpha",
        "Tracking Error (Fund)"
    }

    # Define specific formatting rules for non-percentage metrics (except Up/Down Capture)
    format_dict = {
        "Win/Loss Ratio (Fund)": "{:.3f}",
        "Win/Loss Ratio (Benchmark)": "{:.3f}",
        "Sharpe Ratio (Fund)": "{:.3f}",
        "Sharpe Ratio (Benchmark)": "{:.3f}",
        "Beta": "{:.3f}",
        "Information Ratio": "{:.3f}"
    }

    # Use the row's index (i.e. the metric name) to decide formatting for that entire row.
    def format_row(row):
        metric_name = row.name  # e.g., "Up Capture"
        if metric_name in percentage_metrics:
            return row.apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        elif metric_name in {"Up Capture", "Down Capture"}:
            # Multiply by 100 so that 1.04 shows as 104%
            return row.apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        elif metric_name in format_dict:
            return row.apply(lambda x: format_dict[metric_name].format(x) if pd.notna(x) else "N/A")
        else:
            return row.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    

    
    
    # --- Add a disclaimer with the date up to which the fund returns (and thus the metrics) are calculated ---
    one_year_end = fund_returns.index.max()
    one_year_start = one_year_end - pd.DateOffset(months=12)
    three_year_start = one_year_end - pd.DateOffset(months=36)
    five_year_start = one_year_end - pd.DateOffset(months=60)
    
    st.caption("1Y metrics are based on data from {} to {}"
               .format(one_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    st.caption("3Y metrics are based on data from {} to {}"
               .format(three_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    st.caption("5Y metrics are based on data from {} to {}"
               .format(five_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    
    formatted_table = df_table.apply(format_row, axis=1)
    st.table(formatted_table)
    
    csv_data = df_table.to_csv(index=True)
    st.download_button(
        label="Download Periodic Annualized Stats CSV",
        data=csv_data,
        file_name="periodic_annualized_stats.csv",
        mime="text/csv"
    )
    
    
    
    

    
def plot_rolling_volatility(returns: pd.Series, window: int = 12, title: str = "Rolling Annualized Volatility"):
    """
    Plots rolling annualized volatility using a moving window.
    :param returns: A pandas Series indexed by time.
    :param window: The rolling window in periods (e.g. months).
    :param title: Chart title.
    """
    # Calculate rolling standard deviation (monthly volatility) and annualize it
    rolling_std = returns.rolling(window=window).std() * np.sqrt(12)
    chart_df = pd.DataFrame({"time": returns.index, "Rolling Volatility": rolling_std})
    
    volatility_chart = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("Rolling Volatility:Q", title="Annualized Volatility")
    ).properties(title=title)
    
    
    st.altair_chart(volatility_chart, use_container_width=True)
    
def plot_return_distribution(returns: pd.Series, asset_label: str):
    """
    Plots the distribution of returns for a single asset.
    :param returns: A pandas Series of returns.
    :param asset_label: A label for the asset.
    """
    chart_df = pd.DataFrame({"Returns": returns})
    hist = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Returns:Q", bin=alt.Bin(maxbins=30), title="Return"),
        y=alt.Y("count()", title="Frequency")
    ).properties(title=f"Return Distribution: {asset_label}")
    
    st.altair_chart(hist, use_container_width=True)
def plot_correlation_heatmap(aligned_returns: pd.DataFrame):
    """
    Plots a heatmap for the correlation matrix of the asset returns.
    :param aligned_returns: A DataFrame where each column is an asset's returns.
    """
    corr = aligned_returns.corr()
    corr = corr.reset_index().melt("index")
    heatmap = alt.Chart(corr).mark_rect().encode(
        x=alt.X("variable:N", title="Asset"),
        y=alt.Y("index:N", title="Asset"),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=[alt.Tooltip("index:N"), alt.Tooltip("variable:N"), alt.Tooltip("value:Q", format=".2f")]
    ).properties(title="Correlation Heatmap")
    
    st.altair_chart(heatmap, use_container_width=True)

def plot_risk_return_scatter(annual_metrics_dict: Dict[str, Dict[str, float]], portfolio_metrics: Dict[str, float]):
    """
    Plots a scatter chart comparing annual return and annual volatility.
    :param annual_metrics_dict: Dictionary for individual assets (key = asset name) with keys
                                "Arithmetic Mean Return (Annual)" and "Volatility (Annual)".
    :param portfolio_metrics: Dictionary of portfolio annual metrics (must contain same keys).
    """
    data = []
    for asset, metrics in annual_metrics_dict.items():
        data.append({
            "Asset": asset,
            "Annual Return": metrics.get("Arithmetic Mean Return (Annual)", 0),
            "Volatility": metrics.get("Volatility (Annual)", 0)
        })
    # Add portfolio point
    data.append({
        "Asset": "Portfolio",
        "Annual Return": portfolio_metrics.get("Arithmetic Mean Return (Annual)", 0),
        "Volatility": portfolio_metrics.get("Volatility (Annual)", 0)
    })
    df_scatter = pd.DataFrame(data)
    
    scatter = alt.Chart(df_scatter).mark_circle(size=100).encode(
        x=alt.X("Volatility:Q", title="Annual Volatility"),
        y=alt.Y("Annual Return:Q", title="Annual Return"),
        color=alt.Color("Asset:N"),
        tooltip=["Asset", alt.Tooltip("Annual Return:Q", format=".2%"), alt.Tooltip("Volatility:Q", format=".2%")]
    ).properties(title="Risk vs. Return")
    
    st.altair_chart(scatter, use_container_width=True)

def monte_carlo_simulation_portfolio(aligned_returns: pd.DataFrame, weights: list, horizon: int = 12, n_sims: int = 1000) -> Tuple[pd.DataFrame, np.ndarray, Tuple[int, int]]:
    """
    Perform a robust Monte Carlo simulation for a multi-asset portfolio.
    
    This function simulates each asset’s monthly returns using a multivariate normal distribution,
    computes cumulative returns for each asset, then combines them (using the given weights) to obtain
    the portfolio’s cumulative value.
    
    Parameters:
    -----------
    aligned_returns : pd.DataFrame
        Historical return data for the assets (each column represents an asset).
    weights : list of float
        Portfolio weights for each asset (should sum to 1).
    horizon : int, optional
        Forecast horizon in months. Default is 12.
    n_sims : int, optional
        Number of simulation runs. Default is 1000.
        
    Returns:
    --------
    sim_df : pd.DataFrame
        A DataFrame of simulated cumulative portfolio returns. Each row is a simulation,
        and each column corresponds to a forecast month (named "Month_1", "Month_2", …).
    avg_sim : np.ndarray
        The average simulated portfolio cumulative value for each forecast month.
    extreme_indices : tuple
        A tuple (worst_index, best_index) giving the simulation row indices with the worst and best final values.
    """
    if aligned_returns.empty:
        raise ValueError("The aligned_returns DataFrame is empty. Cannot perform simulation.")
    if len(weights) != aligned_returns.shape[1]:
        raise ValueError("The number of weights must match the number of assets (columns in aligned_returns).")
    
    # Compute historical means and covariance matrix.
    mean_vector = aligned_returns.mean().values
    cov_matrix = aligned_returns.cov().values

    try:
        # Generate simulations for all assets simultaneously.
        # Shape: (n_sims, horizon, num_assets)
        simulations = np.random.multivariate_normal(mean_vector, cov_matrix, size=(n_sims, horizon))
    except Exception as e:
        raise ValueError(f"Error generating simulations: {e}")
    
    # Compute cumulative returns for each asset separately.
    cum_asset_list = []
    num_assets = len(weights)
    for i in range(num_assets):
        asset_sim = simulations[:, :, i]  # shape: (n_sims, horizon)
        cum_asset = (1 + asset_sim).cumprod(axis=1)
        cum_asset_list.append(cum_asset)
    
    # Compute portfolio cumulative value as the weighted sum of the individual asset cumulative returns.
    portfolio_cum_values = np.zeros((n_sims, horizon))
    for i, w in enumerate(weights):
        portfolio_cum_values += w * cum_asset_list[i]
    
    # Calculate the average simulation (across all simulations) for each month.
    avg_sim = portfolio_cum_values.mean(axis=0)
    
    # Identify the best and worst simulation by final portfolio value.
    final_values = portfolio_cum_values[:, -1]
    best_index = int(np.argmax(final_values))
    worst_index = int(np.argmin(final_values))
    
    # Create a DataFrame of simulated cumulative portfolio returns.
    sim_df = pd.DataFrame(portfolio_cum_values, columns=[f"Month_{i+1}" for i in range(horizon)])
    
    return sim_df, avg_sim, (worst_index, best_index)

 
# -------------------------------------------------------------------
# Function to Compute Simulation Summary Statistics
# -------------------------------------------------------------------
def compute_simulation_stats(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics based on the simulation final portfolio values.
    
    The statistics include:
      - Mean Final Value
      - Median Final Value
      - Worst Final Value
      - Best Final Value
      - Mean Growth (final value minus 1)
      - 95% VaR (5th percentile of final values minus 1)
    
    Returns:
    --------
    stats_df : pd.DataFrame
        A DataFrame with the computed statistics.
    """
    final_vals = sim_df.iloc[:, -1]
    mean_final = final_vals.mean() 
    median_final = final_vals.median()
    worst_final = final_vals.min()
    best_final = final_vals.max()
    mean_growth = mean_final - 1
    var_95 = np.percentile(final_vals, 5) - 1  # 5th percentile as VaR at 95% confidence
    stats_df = pd.DataFrame({
        "Statistic": ["Mean Final Value", "Median Final Value", "Worst Final Value", "Best Final Value", "Mean Growth", "95% VaR"],
        "Value": [mean_final, median_final, worst_final, best_final, mean_growth, var_95]
    })
    return stats_df

def maximize_calmar_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0, allow_short: bool = False) -> np.ndarray:
    """
    Optimize portfolio weights to maximize the Calmar Ratio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical return data with each column representing an asset.
    risk_free_rate : float
        The risk-free rate (in the same periodic units as the returns, not used directly here but kept for consistency).
    allow_short : bool
        If False, weights are constrained to be non-negative (no short selling).
        
    Returns:
    --------
    np.ndarray
        The optimized weights that maximize the Calmar Ratio (summing to 1).
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values  # Monthly mean returns

    def portfolio_performance(weights):
        # Portfolio monthly returns as a NumPy array
        port_returns = np.dot(returns, weights)
        # Convert to pandas Series for cumulative operations
        port_series = pd.Series(port_returns, index=returns.index)
        # Annualized return
        annual_return = (1 + port_series.mean()) ** 12 - 1
        # Cumulative returns for drawdown calculation
        cum_returns = (1 + port_series).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = abs(drawdown.min())  # Absolute value of max drawdown
        return annual_return, max_drawdown

    # The negative Calmar Ratio (since we minimize)
    def neg_calmar_ratio(weights):
        annual_return, max_drawdown = portfolio_performance(weights)
        calmar = annual_return / max_drawdown if max_drawdown > 0 else -np.inf  # Avoid division by zero
        return -calmar  # Minimize negative to maximize Calmar

    # Constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    # Bounds: non-negative weights if no short selling is allowed
    bounds = None if allow_short else tuple((0, 1) for _ in range(n))
    
    # Initial guess: equal allocation
    init_guess = np.repeat(1/n, n)
    
    result = minimize(neg_calmar_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    return result.x

def maximize_sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0, allow_short: bool = False) -> np.ndarray:
    """
    Optimize portfolio weights to maximize the Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical return data with each column representing an asset.
    risk_free_rate : float
        The risk-free rate (in the same periodic units as the returns).
    allow_short : bool
        If False, weights are constrained to be non-negative (no short selling).
        
    Returns:
    --------
    np.ndarray
        The optimized weights that maximize the Sharpe ratio (summing to 1).
    """
    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return, port_vol

    # The negative Sharpe ratio (since we minimize)
    def neg_sharpe_ratio(weights):
        port_return, port_vol = portfolio_performance(weights)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        return -sharpe

    # Constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    # Bounds: non-negative weights if no short selling is allowed
    bounds = None if allow_short else tuple((0, 1) for _ in range(n))
    
    # Initial guess: equal allocation
    init_guess = np.repeat(1/n, n)
    
    result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    return result.x

def portfolio_drift_rebalancing_simulation(asset_data: List[pd.DataFrame], asset_names: List[str], weights: List[float]):
    st.subheader("Portfolio Drift & Rebalancing Simulation")
    st.write("""
    This simulation shows how your portfolio would evolve if allowed to drift and then rebalanced at different frequencies.
    Choose the rebalancing frequency from the options below.
    """)

    # Create a toggle (radio buttons) for rebalancing frequency.
    rebalance_option = st.radio(
        "Select Rebalancing Frequency",
        ("No Rebalancing", "Rebalance Every 6 months", "Rebalance Every 1 year", "Rebalance Every 2 Years")
    )
    
    # Determine the rebalancing period in months.
    if rebalance_option == "No Rebalancing":
        rebalance_period = None  # No rebalancing
    elif rebalance_option == "Rebalance Every 6 months":
        rebalance_period = 6
    elif rebalance_option == "Rebalance Every 1 year":
        rebalance_period = 12
    elif rebalance_option == "Rebalance Every 2 Years":
        rebalance_period = 24
    else:
        rebalance_period = None

    # Run the simulation. Note that the simulation function now returns both the simulation DataFrame and final weights.
    sim_df, final_weights = simulate_rebalanced_portfolio(asset_data, asset_names, weights, rebalance_period)
    
    if sim_df.empty:
        st.error("Simulation failed: no data available.")
        return

    # Plot the simulated portfolio value over time.
    sim_chart = alt.Chart(sim_df).mark_line().encode(
        x=alt.X('time:T', title="Time"),
        y=alt.Y('portfolio_value:Q', title="Portfolio Value"),
        tooltip=[alt.Tooltip('time:T', title='Date'),
                 alt.Tooltip('portfolio_value:Q', title='Portfolio Value', format=".4f")]
    ).properties(
        title="Simulated Portfolio Growth with Drift & Rebalancing"
    )
    
    st.altair_chart(sim_chart, use_container_width=True)
    
    
    # Create a DataFrame for the final portfolio weights.
    final_weights_df = pd.DataFrame({
    "Asset": asset_names,
    "Original Weight (%)": (np.array(weights) * 100).round(2),
    "Final Weight (%)": (final_weights * 100).round(2)
    })
    
    st.write("**Final Portfolio Weights**")
    st.table(final_weights_df)


def simulate_rebalanced_portfolio(asset_data: List[pd.DataFrame], asset_names: List[str], 
                                  weights: List[float], rebalance_period: int = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate portfolio growth using individual asset return data, allowing the portfolio weights
    to drift over time and rebalancing them at a given frequency.

    Returns:
        result_df (pd.DataFrame): DataFrame containing the time series of portfolio value.
        final_weights (np.ndarray): Array of final portfolio weights (as proportions).
    """
    # 1. Determine the common dates across all asset data.
    date_sets = [set(df['time'].dropna().unique()) for df in asset_data if not df.empty]
    if not date_sets:
        st.error("No valid asset data available for simulation.")
        return pd.DataFrame(), np.array([])
    common_dates = sorted(set.intersection(*date_sets))
    if not common_dates:
        st.error("No overlapping dates found across assets.")
        return pd.DataFrame(), np.array([])

    # Convert the list of dates to a DatetimeIndex for our simulation DataFrame.
    sim_index = pd.to_datetime(common_dates)
    
    # 2. Build a DataFrame containing each asset's total_return series, reindexed on the common dates.
    asset_return_df = pd.DataFrame(index=sim_index)
    for i, df in enumerate(asset_data):
        if df.empty:
            continue
        temp = df.copy()
        temp['time'] = pd.to_datetime(temp['time'])
        temp = temp.set_index('time').sort_index()
        asset_return_df[asset_names[i]] = temp['total_return'].reindex(sim_index).fillna(method='ffill')
    
    # 3. Simulation:
    n_periods = len(asset_return_df)
    portfolio_values = []   # To store the portfolio total value over time
    current_values = np.array(weights, dtype=float)  # Starting asset amounts (each is weight * 1)
    portfolio_values.append(current_values.sum())  # initial total (should equal 1)

    # Loop over each period to update asset values.
    for t in range(1, n_periods):
        period_returns = asset_return_df.iloc[t].values
        current_values = current_values * (1 + period_returns)
        total_value = current_values.sum()
        # Rebalance if a rebalancing period is set and the period is a multiple of that interval.
        if rebalance_period and (t % rebalance_period == 0):
            current_values = total_value * np.array(weights)
        portfolio_values.append(total_value)
    
    # Build the DataFrame for plotting.
    result_df = pd.DataFrame({
        'time': sim_index,
        'portfolio_value': portfolio_values
    })
    
    # Compute final portfolio weights (as proportions).
    final_weights = current_values / current_values.sum()
    
    return result_df, final_weights


def plot_annual_returns_over_time(asset_data: List[pd.DataFrame], asset_names: List[str]) -> None:
    """
    Computes the annual return for each asset per year and displays a heatmap.
    Assets are on the Y-axis, years on the X-axis, with color indicating return magnitude.
    
    For each asset, the annual return for a given year is calculated as:
        Annual Return = (∏ (1 + monthly return)) - 1
    """
    rows = []
    for asset, df in zip(asset_names, asset_data):
        if df.empty or 'total_return' not in df.columns:
            continue
        # Ensure the time column is datetime and create a Year column
        df['time'] = pd.to_datetime(df['time'])
        df['Year'] = df['time'].dt.year
        # Group by year and compute the annual return
        grouped = df.groupby('Year')['total_return'].apply(
            lambda x: np.prod(1 + x) - 1
        ).reset_index().rename(columns={'total_return': 'Annual Return'})
        grouped['Asset'] = asset
        rows.append(grouped)
    
    if not rows:
        st.warning("No valid asset data available to compute annual returns.")
        return
    
    data = pd.concat(rows, ignore_index=True)
    
    # Create a heatmap using Altair
    heatmap = alt.Chart(data).mark_rect().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Asset:N', title='Asset'),
        color=alt.Color('Annual Return:Q', 
                        scale=alt.Scale(scheme='redyellowgreen', domainMid=0),  # Center at 0, red for negative, green for positive
                        title='Annual Return'),
        tooltip=[
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('Asset:N', title='Asset'),
            alt.Tooltip('Annual Return:Q', format='.2%', title='Annual Return')
        ]
    ).properties(
        title="Annual Returns Heatmap by Asset",
        width=600,
        height=300
    ).interactive()
    
    st.altair_chart(heatmap, use_container_width=True)

def plot_rolling_annual_returns(portfolio_returns: pd.Series, window: int = 12, 
                                title: str = "Rolling 1Y Annualized Returns") -> None:
    """
    Computes a rolling annualized return for the portfolio and plots it as a line chart.
    
    For each rolling window, the annualized return is computed as:
         (product(1 + monthly returns))^(12 / window) - 1
    """
    rolling_annual = portfolio_returns.rolling(window=window).apply(
        lambda x: np.prod(1 + x) ** (12 / len(x)) - 1, raw=True
    )
    df_rolling = pd.DataFrame({
        "time": portfolio_returns.index,
        "Rolling Annual Return": rolling_annual
    })
    
    chart = alt.Chart(df_rolling).mark_line().encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("Rolling Annual Return:Q", title="Rolling Annualized Return", 
                axis=alt.Axis(format=".2%")),
        tooltip=[alt.Tooltip("time:T", title="Date", format="%Y-%m-%d"),
                 alt.Tooltip("Rolling Annual Return:Q", format=".2%")]
    ).properties(
        title=title
    )
    
    st.altair_chart(chart, use_container_width=True)#########################

def plot_dividend_decomposition(df: pd.DataFrame, asset_name: str) -> None:
    """
    For a given asset dataframe, this function computes the price return and dividend return,
    creates a stacked bar chart showing the breakdown of returns (normalized to 100% per period),
    and displays a table with the average percentage of total return coming from dividends.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The asset data. Must include columns 'time', 'price', 'dividend', and 'total_return'.
    asset_name : str
        A label for the asset.
    """
    # Ensure the data is sorted by time
    df = df.sort_values('time').copy()
    
    # Calculate price return and dividend return (if not already computed)
    df['price_return'] = df['price'].pct_change()
    # Dividend return is the residual from total return
    df['dividend_return'] = df['total_return'] - df['price_return']
    
    # Prepare data for the stacked bar chart: melt the returns into long format
    df_melt = df.melt(id_vars='time', 
                      value_vars=['price_return', 'dividend_return'],
                      var_name='Return Component', 
                      value_name='Return')
    
    # Create a stacked (normalized) bar chart using Altair
    chart = alt.Chart(df_melt).mark_bar().encode(
        x=alt.X('time:T', title='Time'),
        y=alt.Y('sum(Return):Q', title='Return Contribution', stack="normalize"),
        color=alt.Color('Return Component:N', title='Component', 
                        scale=alt.Scale(
                            domain=['price_return', 'dividend_return'],
                            range=['#1f77b4', '#2ca02c']  # Blue for price, green for dividends
                        )),
        tooltip=[
            alt.Tooltip('time:T', title='Date'),
            alt.Tooltip('Return Component:N'),
            alt.Tooltip('sum(Return):Q', title='Contribution', format='.2%')
        ]
    ).properties(
        title=f"Return Decomposition for {asset_name}: Price vs. Dividend"
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Compute the percentage of total return attributable to dividends for each period.
    # (Avoid division by zero by replacing zeros in total_return with np.nan)
    df['dividend_pct'] = np.where(df['total_return'] != 0, 
                                  (df['dividend_return'] / df['total_return']) * 100, 
                                  np.nan)
    
    # Create a summary table that shows the average dividend contribution
    avg_dividend_pct = df['dividend_pct'].mean()
    summary_df = pd.DataFrame({
        'Asset': [asset_name],
        'Average Dividend % of Total Return': [avg_dividend_pct]
    })
    st.subheader(f"{asset_name} – Dividend Contribution Summary")
    st.table(summary_df.style.format({"Average Dividend % of Total Return": "{:.2f}%" }))

#  The Main Function    #
def main():
    st.title("Portfolio Analytics Dashboard")
    
    st.subheader("Download Helper Documentation")
    with open("docs/helper_doc.pdf", "rb") as pdf_file:
        pdf_data = pdf_file.read()
    st.download_button(
        label="Download Helper Documentation (PDF)",
        data=pdf_data,
        file_name="Portfolio_Analytics_Dashboard_Helper_Documentation.pdf",
        mime="application/pdf",
        key="download_helper_doc"
    )
    st.markdown("---")
    
    
    # Define percentage_metrics for use in both tabs
    percentage_metrics = {
        "Ann Return (Fund)", "Ann Return (Benchmark)", "Excess Return", "Ann Risk-free Rate",
        "Ann Std Dev (Fund)", "Ann Std Dev (Benchmark)", "Downside Deviation (Fund)",
        "Downside Deviation (Benchmark)", "Maximum Drawdown (Fund)", "Maximum Drawdown (Benchmark)",
        "Alpha", "Tracking Error (Fund)"
    }
    
    # === Sidebar: Global Configuration & Asset Upload ===
    with st.sidebar:
        st.header("Configuration")
        num_assets = st.number_input(
            "**Number of Assets**", min_value=1, max_value=20, value=1, help="Select the number of assets in your portfolio."
        )
        dividend_as_yield = st.radio(
            "**Dividend Input Type**", ("Yield", "Actual Amount"), index=0, help="Specify whether dividends are provided as a yield (%) or actual amounts."
        ) == "Yield"
        st.markdown("---")
        st.header("Asset Upload")
        asset_data = []
        asset_names = []
        asset_benchmarks = []
        for i in range(num_assets):
            
            with st.expander(f"Asset {i+1}"):
                asset_name = st.text_input(f"Asset {i+1} Name", f"Asset {i+1}", key=f"name_{i}", help="Provide a name for the asset.")
                uploaded_file = st.file_uploader(f"Upload CSV for {asset_name}", type=["csv"], key=f"file_{i}", help="Upload a CSV file containing the asset's data.")
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        df = adjust_column_names(df)
                        if (('price' not in df.columns) or (df.get('price').isnull().all())) and ('return' in df.columns):
                            df['price'] = (1 + df['return']).cumprod()
                        elif 'price' in df.columns and ('return' not in df.columns):
                            df['return'] = df['price'].pct_change()
                        validate_columns(df)
                        df['time'] = pd.to_datetime(df['time']).dt.to_period('M').dt.to_timestamp('M')
                        if not risk_free_df.empty:
                            df = pd.merge(df, risk_free_df, on='time', how='left')
                        df, _, _ = calculate_metrics(df, dividend_as_yield)
                        asset_data.append(df)
                        asset_names.append(asset_name)
                    except Exception as e:
                        st.error(f"Error processing {asset_name}: {str(e)}")
                        asset_data.append(pd.DataFrame())
                        asset_names.append(asset_name)
                else:
                    st.warning(f"No file uploaded for {asset_name}. Adding empty data for this asset.")
                    asset_data.append(pd.DataFrame())
                    asset_names.append(asset_name)
                benchmark_type = st.radio(f"Benchmark Source for {asset_name}", ["None", "Ticker", "CSV"], key=f"bench_type_{i}")
                if benchmark_type == "Ticker":
                    bench_ticker = st.text_input(f"Benchmark Ticker for {asset_name}", "", key=f"bench_ticker_{i}")
                    if bench_ticker and not asset_data[-1].empty:
                        start_date, end_date = validate_time_ranges([asset_data[-1]])
                        bench_df = fetch_benchmark_data(bench_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                        asset_benchmarks.append(bench_df)
                    else:
                        asset_benchmarks.append(pd.DataFrame())
                elif benchmark_type == "CSV":
                    bench_file = st.file_uploader(f"Upload Benchmark CSV for {asset_name}", type=["csv"], key=f"bench_file_{i}")
                    if bench_file:
                        bench_df = pd.read_csv(bench_file)
                        bench_df = adjust_column_names(bench_df)
                        bench_df['time'] = pd.to_datetime(bench_df['time']).dt.to_period('M').dt.to_timestamp('M')
                        if 'return' not in bench_df.columns and 'price' in bench_df.columns:
                            bench_df['return'] = bench_df['price'].pct_change()
                        asset_benchmarks.append(bench_df)
                    else:
                        asset_benchmarks.append(pd.DataFrame())
                else:
                    asset_benchmarks.append(pd.DataFrame())
        
        st.markdown("---")
        st.header("Portfolio Configuration")
        asset_allocations = []
        total_allocation = 0.0
        for asset in asset_names:
            allocation = st.number_input(
                f"Allocation (%) for {asset}", min_value=0.0, max_value=100.0, value=round(100/num_assets, 2), 
                step=0.01, format="%.2f", key=f"alloc_{asset}"
            )
            asset_allocations.append(allocation)
            total_allocation += allocation
        st.write(f"**Total Allocation:** {total_allocation:.2f}%")
        if not np.isclose(total_allocation, 100.0, atol=0.01):
            st.error("Error: The total allocation of all assets must equal 100%.")
            st.stop()
        asset_weights = normalize_weights(asset_allocations)
        

    # Fetch benchmark data
    if benchmark_source == "Fetch from Ticker":
        if benchmark_ticker and all(not df.empty for df in asset_data):
            try:
                start_date, end_date = validate_time_ranges(asset_data)
                benchmark_data = fetch_benchmark_data(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            except ValueError as e:
                st.error(f"Time range validation failed: {e}")
                return
        else:
            benchmark_data = pd.DataFrame()
    
    else:  # Upload Benchmark CSV
        if benchmark_file is not None:
            try:
                benchmark_data = pd.read_csv(benchmark_file)
                benchmark_data = adjust_column_names(benchmark_data)  # Standardize column names
                # Ensure required columns and calculate return if missing
                if ('price' not in benchmark_data.columns or benchmark_data['price'].isnull().all()) and ('return' in benchmark_data.columns):
                    benchmark_data['price'] = (1 + benchmark_data['return']).cumprod()
                elif 'price' in benchmark_data.columns and 'return' not in benchmark_data.columns:
                    benchmark_data['return'] = benchmark_data['price'].pct_change()
                else:
                    raise ValueError("Either 'price' or 'return' must be provided in the benchmark CSV.")
                # Ensure 'dividend' exists, default to 0 if missing
                if 'dividend' not in benchmark_data.columns:
                    benchmark_data['dividend'] = 0
                validate_columns(benchmark_data)  # Validate required columns
                benchmark_data['time'] = pd.to_datetime(benchmark_data['time']).dt.to_period('M').dt.to_timestamp('M')
                # Drop any rows with NaN in critical columns
                benchmark_data = benchmark_data.dropna(subset=['time', 'price', 'return'])
                # Log for debugging
                st.write("DEBUG: Processed Benchmark Data from CSV", benchmark_data.head())
            except Exception as e:
                st.error(f"Error processing benchmark CSV: {str(e)}. Ensure the file has columns 'time', 'price', 'dividend', and 'return' (or compute 'return' from 'price').")
                benchmark_data = pd.DataFrame()
        else:
            benchmark_data = pd.DataFrame()

    # Main UI with tabs
    tab1, tab2 = st.tabs(["Portfolio Analysis", "Individual Asset Metrics"])
    
    if all(not df.empty for df in asset_data):
        try:
            start_date, end_date = validate_time_ranges(asset_data)
            if benchmark_source == "Fetch from Ticker":
                benchmark_data = fetch_benchmark_data(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            total_weight = sum(asset_weights)
            if total_weight > 0:
                asset_weights = [w / total_weight for w in asset_weights]
            
            portfolio_rets, raw_port_metrics, annual_port_metrics, risk_decomp_table, weighted_benchmark_returns = portfolio_metrics(
                asset_data=asset_data, weights=asset_weights, asset_names=asset_names,
                benchmark_data=(benchmark_data if use_global_benchmark and not benchmark_data.empty else None),
                risk_free_df=(risk_free_df if not risk_free_df.empty else None),
                asset_benchmarks=(asset_benchmarks if not use_global_benchmark else None)
            )
            benchmark_returns = weighted_benchmark_returns
            asset_benchmarks_dict = {}
            if not use_global_benchmark:
                for name, bench_df in zip(asset_names, asset_benchmarks):
                    asset_benchmarks_dict[name] = bench_df.set_index('time')['return'].reindex(portfolio_rets.index, method='ffill') if not bench_df.empty else pd.Series(0.0, index=portfolio_rets.index)
            else:
                for name in asset_names:
                    asset_benchmarks_dict[name] = benchmark_returns
            risk_free_series = risk_free_df.set_index('time')['risk_free_rate'].reindex(portfolio_rets.index, method='ffill') if not risk_free_df.empty else pd.Series(0.0, index=portfolio_rets.index)
            common_index = portfolio_rets.index.intersection(benchmark_returns.index).intersection(risk_free_series.index)
            portfolio_rets = portfolio_rets.reindex(common_index)
            benchmark_returns = benchmark_returns.reindex(common_index)
            risk_free_series = risk_free_series.reindex(common_index, method='ffill')
            asset_returns_dict = {name: df.set_index('time')['total_return'].reindex(common_index) for name, df in zip(asset_names, asset_data) if not df.empty}
            aligned_returns = align_asset_data(asset_data, asset_names)

            with tab1:  # Portfolio Analysis
                with st.expander("Periodic Annualized Stats (1Y, 3Y, 5Y, ITD)", expanded=False):
                    date_range = st.date_input("Select Date Range", [portfolio_rets.index.min().date(), portfolio_rets.index.max().date()])
                    st.write("The portfolio benchmark is calculated using the weighted returns of your asset benchmarks. For example, with a 60/40 allocation, returns from each benchmark are weighted at 0.6 and 0.4, respectively.")
                    
                    if len(date_range) == 2 and date_range[0] <= date_range[1]:
                        filtered_rets = portfolio_rets[(portfolio_rets.index >= pd.to_datetime(date_range[0])) & (portfolio_rets.index <= pd.to_datetime(date_range[1]))]
                        filtered_bench = benchmark_returns[(benchmark_returns.index >= pd.to_datetime(date_range[0])) & (benchmark_returns.index <= pd.to_datetime(date_range[1]))]
                        filtered_risk = risk_free_series[(risk_free_series.index >= pd.to_datetime(date_range[0])) & (risk_free_series.index <= pd.to_datetime(date_range[1]))]
                        display_full_periodic_table(filtered_rets, filtered_bench, filtered_risk)

                with st.expander("Portfolio Performance & Rebalancing Simulation", expanded=False):
                    st.subheader("Portfolio Growth vs. Benchmark Growth")
                    st.write("This graph illustrates the portfolio's growth over time compared to a benchmark")
                    # Single date range control for both Portfolio Growth and Rebalancing Simulation
                    growth_date_range = st.date_input(
                        "Select Date Range for Portfolio Growth and Rebalancing Simulation",
                        [portfolio_rets.index.min().date(), portfolio_rets.index.max().date()],
                        key="growth_and_rebalance_date"
                    )
                    
                    if len(growth_date_range) == 2 and growth_date_range[0] <= growth_date_range[1]:
                        filtered_rets = portfolio_rets[
                            (portfolio_rets.index >= pd.to_datetime(growth_date_range[0])) & 
                            (portfolio_rets.index <= pd.to_datetime(growth_date_range[1]))
                        ]
                        filtered_bench = benchmark_returns[
                            (benchmark_returns.index >= pd.to_datetime(growth_date_range[0])) & 
                            (benchmark_returns.index <= pd.to_datetime(growth_date_range[1]))
                        ]
                        
                        # Portfolio Growth Chart
                        if len(asset_data) > 1:
                            growth_df = pd.DataFrame({
                                'time': filtered_rets.index, 
                                'Growth': (1 + filtered_rets).cumprod(), 
                                'Label': 'Portfolio'
                            }).dropna(subset=['time']).sort_values('time')
                            if not filtered_bench.empty:
                                bench_df = pd.DataFrame({
                                    'time': filtered_bench.index, 
                                    'Growth': (1 + filtered_bench).cumprod(), 
                                    'Label': 'Benchmark'
                                }).dropna(subset=['time']).sort_values('time')
                                combined_df = pd.concat([growth_df, bench_df], ignore_index=True)
                                chart = alt.Chart(combined_df).mark_line().encode(
                                    x='time:T',
                                    y='Growth:Q',
                                    color=alt.Color('Label:N', scale=alt.Scale(
                                        domain=['Portfolio', 'Benchmark'],
                                        range=['blue', 'grey']  # Portfolio: blue, Benchmark: grey
                                    )),
                                    tooltip=['time:T', 'Label:N', 'Growth:Q']
                                ).properties(title="Portfolio vs Benchmark Growth").interactive()
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                chart = alt.Chart(growth_df).mark_line(color='blue').encode(
                                    x='time:T',
                                    y='Growth:Q',
                                    tooltip=['time:T', 'Growth:Q']
                                ).properties(title="Portfolio Growth").interactive()
                                st.altair_chart(chart, use_container_width=True)
                        
                        # Portfolio Drift & Rebalancing Simulation
                        filtered_asset_data = [
                            df[(df['time'] >= pd.to_datetime(growth_date_range[0])) & 
                            (df['time'] <= pd.to_datetime(growth_date_range[1]))]
                            for df in asset_data
                        ]
                        portfolio_drift_rebalancing_simulation(filtered_asset_data, asset_names, asset_weights)

                with st.expander("Monthly Performance Table", expanded=False):
                    st.subheader("Monthly Performance Table")
                    st.write("This table presents the monthly performance of the portfolio compared to its weighted benchmark. It provides a month-by-month breakdown of returns, allowing for direct comparison between actual portfolio performance and the benchmark’s expected performance.")
                    if not portfolio_rets.empty and not benchmark_returns.empty:
                        # Resample portfolio returns to yearly totals
                        portfolio_returns_yearly = portfolio_rets.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                        # Resample weighted benchmark returns (Series) to yearly totals
                        benchmark_yearly = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                        portfolio_returns_yearly.index = portfolio_returns_yearly.index.year
                        benchmark_yearly.index = benchmark_yearly.index.year
                        
                        # Prepare monthly table for portfolio returns
                        portfolio_monthly = portfolio_rets.reset_index()
                        portfolio_monthly.columns = ['time', 'return']
                        portfolio_monthly['Year'] = portfolio_monthly['time'].dt.year
                        portfolio_monthly['Month'] = portfolio_monthly['time'].dt.month
                        monthly_table = portfolio_monthly.pivot(index='Year', columns='Month', values='return')
                        
                        # Ensure all months are included (1-12)
                        all_months = pd.MultiIndex.from_product([monthly_table.index, range(1, 13)], names=['Year', 'Month'])
                        monthly_table = portfolio_monthly.set_index(['Year', 'Month']).reindex(all_months)['return'].unstack()
                        
                        # Add yearly totals for portfolio and benchmark
                        monthly_table['Total'] = portfolio_returns_yearly.reindex(monthly_table.index, fill_value=np.nan)
                        monthly_table['Benchmark'] = benchmark_yearly.reindex(monthly_table.index, fill_value=np.nan)
                        
                        # Rename months to names
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        monthly_table.rename(columns={i + 1: month_names[i] for i in range(12)}, inplace=True)
                        monthly_table.index = monthly_table.index.map(str)
                        
                        # Display table with formatting
                        st.dataframe(monthly_table.style.format("{:.2%}"))
                        csv_data = monthly_table.to_csv(index=True)
                        st.download_button(label="Download Monthly Performance CSV", data=csv_data, file_name="monthly_performance.csv", mime="text/csv")
                        
                        # Display best and worst years
                        best_year, best_return = portfolio_returns_yearly.idxmax(), portfolio_returns_yearly.max()
                        worst_year, worst_return = portfolio_returns_yearly.idxmin(), portfolio_returns_yearly.min()
                        st.markdown(f"**Best Year:** {best_year} ({best_return:.2%})")
                        st.markdown(f"**Worst Year:** {worst_year} ({worst_return:.2%})")
                    else:
                        st.warning("Portfolio or benchmark data is empty. Cannot display monthly performance table.")

                with st.expander("Risk Analysis", expanded=False):
                    if not risk_decomp_table.empty:
                        risk_decomp_df = risk_decomp_table.reset_index().rename(columns={'index': 'Asset'})
                        pie_chart = alt.Chart(risk_decomp_df).mark_arc().encode(
                            theta='Percent of Risk:Q', color='Asset:N', tooltip=['Asset', 'Percent of Risk']
                        ).properties(title="Risk Decomposition")
                        st.subheader("Risk Decomposition Pie Chart")
                        st.write("This pie chart illustrates the contribution of each asset to the overall portfolio risk. Rather than just showing allocation weights, it highlights which assets contribute the most to portfolio volatility.")
                        st.altair_chart(pie_chart, use_container_width=True)
                    st.subheader("Drawdowns of Assets & Portfolio Over Time")
                    st.write("This graph visualizes the historical drawdowns of individual assets and the overall portfolio. A drawdown represents the percentage decline from the portfolio's previous peak, helping to measure downside risk. Click on the Labels to view Individual Breakdown")
                    plot_drawdown_of_assets(asset_data, asset_names, portfolio_rets)
                    growth_df = pd.DataFrame({'time': portfolio_rets.index, 'Growth': (1 + portfolio_rets).cumprod(), 'Label': 'Portfolio'})
                    bench_df = pd.DataFrame({'time': benchmark_returns.index, 'Growth': (1 + benchmark_returns).cumprod(), 'Label': 'Benchmark'})
                    combined_df = pd.concat([growth_df, bench_df], ignore_index=True)
                    chart = alt.Chart(combined_df).mark_line().encode(
                        x='time:T', 
                        y='Growth:Q', 
                        color=alt.Color('Label:N', scale=alt.Scale(
                            domain=['Portfolio', 'Benchmark'],
                            range=['blue', 'grey']
                        )), 
                        tooltip=['time:T', 'Label:N', 'Growth:Q']
                    ).properties(title="Portfolio vs Benchmark Growth").interactive()
                

                with st.expander("Advanced Risk/Return Metrics", expanded=False):
                    
                    plot_rolling_volatility(portfolio_rets)
                    plot_annual_returns_over_time(asset_data, asset_names)
                    plot_rolling_annual_returns(portfolio_rets)
                    
                    if not aligned_returns.empty:
                        st.write("**Correlation HeatMap**")
                        corr_df = aligned_returns.corr()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu", center=0, ax=ax)
                        st.pyplot(fig)

                with st.expander("Monte Carlo Simulation", expanded=False):
                    st.subheader("Monte Carlo Simulation")
                    st.write("This chart shows a Monte Carlo simulation, which forecasts how your portfolio might perform in the future based on 1,000 different simulated outcomes. It's like running a "'what-if'" experiment a thousand times to see what might happen to your investments.")
                    sim_horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=60, value=12, step=1)
                    n_sims = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
                    try:
                        sim_df, avg_sim, (worst_idx, best_idx) = monte_carlo_simulation_portfolio(aligned_returns, asset_weights, horizon=sim_horizon, n_sims=n_sims)
                        sim_df_reset = sim_df.reset_index(drop=True)
                        sim_df_reset["Simulation"] = sim_df_reset.index.astype(str)
                        sim_long = pd.melt(sim_df_reset, id_vars=["Simulation"], var_name="Month", value_name="Value")
                        sim_long["Month"] = sim_long["Month"].str.replace("Month_", "").astype(int)
                        base = alt.Chart(sim_long).mark_line(color='lightgray', opacity=0.2).encode(x="Month:Q", y="Value:Q", detail="Simulation:N")
                        avg_df = pd.DataFrame({"Month": np.arange(1, sim_horizon + 1), "Value": avg_sim})
                        avg_line = alt.Chart(avg_df).mark_line(color='black', strokeWidth=3).encode(x="Month:Q", y="Value:Q")
                        best_line = alt.Chart(sim_long[sim_long["Simulation"] == str(best_idx)]).mark_line(color='green', strokeWidth=3).encode(x="Month:Q", y="Value:Q")
                        worst_line = alt.Chart(sim_long[sim_long["Simulation"] == str(worst_idx)]).mark_line(color='red', strokeWidth=3).encode(x="Month:Q", y="Value:Q")
                        chart = alt.layer(base, avg_line, best_line, worst_line).properties(title="Monte Carlo Simulation")
                        st.altair_chart(chart, use_container_width=True)
                        stats_df = compute_simulation_stats(sim_df)
                        st.write("""
                        This table presents key insights from the Monte Carlo simulation, which models potential future portfolio performance over multiple scenarios. It provides a statistical overview of expected returns, risks, and possible best- and worst-case outcomes.
                        """)
                        st.dataframe(stats_df.style.format({"Value": "{:.2%}"}))
                        csv_data = stats_df.to_csv(index=False)
                        st.download_button(label="Download Simulation Stats CSV", data=csv_data, file_name="monte_carlo_stats.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Monte Carlo simulation error: {e}")

                with st.expander("Sharpe Optimized Portfolio Allocation", expanded=False):
                    st.write("""
                    This section optimizes your portfolio to maximize the Sharpe Ratio, which measures how much return you get for each unit of risk. The optimization tweaks your asset weights to find the best balance between higher returns and lower volatility, assuming no short selling (all weights stay positive). The chart compares your current portfolio’s growth (blue) with this optimized version (green), showing how adjusting your allocations could potentially boost performance while keeping risk in check.
                    """)
                    if not aligned_returns.empty:
                        try:
                            optimized_weights = maximize_sharpe_ratio(aligned_returns, risk_free_rate=0.001, allow_short=False)
                            weights_df = pd.DataFrame({"Asset": asset_names, "Current Weight": asset_weights, "Optimized Weight": optimized_weights})
                            st.dataframe(weights_df)

                            # Optimized portfolio returns
                            opt_rets = aligned_returns.dot(optimized_weights)
                            opt_growth = pd.DataFrame({
                                'time': aligned_returns.index,
                                'Growth': (1 + opt_rets).cumprod(),
                                'Label': 'Optimized Portfolio'
                            })

                            # Current portfolio returns (using original weights)
                            current_rets = aligned_returns.dot(asset_weights)
                            current_growth = pd.DataFrame({
                                'time': aligned_returns.index,
                                'Growth': (1 + current_rets).cumprod(),
                                'Label': 'Current Portfolio'
                            })

                            # Combine data for plotting
                            combined_growth = pd.concat([current_growth, opt_growth], ignore_index=True)

                            # Plot both lines using Altair
                            chart = alt.Chart(combined_growth).mark_line().encode(
                                x=alt.X('time:T', title='Time'),
                                y=alt.Y('Growth:Q', title='Cumulative Growth'),
                                color=alt.Color('Label:N', scale=alt.Scale(
                                    domain=['Current Portfolio', 'Sharpe-Optimizedd Portfolio'],
                                    range=['blue', 'green']  # Blue for current, green for optimized
                                )),
                                tooltip=[alt.Tooltip('time:T', title='Date', format='%Y-%m-%d'),
                                         alt.Tooltip('Label:N', title='Portfolio'),
                                         alt.Tooltip('Growth:Q', title='Growth', format='.4f')]
                            ).properties(
                                title="Current vs Optimized Portfolio Growth",
                                width=600,
                                height=400
                            ).interactive()

                            st.altair_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Optimization error: {e}")
                            
                with st.expander("Drawdown-Optimized Growth (Max Calmar Ratio)", expanded=False):
                    st.write("""
                    This optimization focuses on maximizing the Calmar Ratio, which looks at your portfolio’s annual return relative to its worst drop (maximum drawdown). It adjusts your asset weights to prioritize steady growth while minimizing big losses, without allowing short selling (all weights stay positive). The chart compares your current portfolio (blue) to this drawdown-optimized version (green), highlighting how these new weights could help protect against steep declines while still aiming for solid returns. It’s ideal if you want to sleep better during market downturns.
                    """)
                    if not aligned_returns.empty:
                        try:
                            calmar_weights = maximize_calmar_ratio(aligned_returns, risk_free_rate=0.001, allow_short=False)
                            weights_df = pd.DataFrame({
                                "Asset": asset_names,
                                "Current Weight": asset_weights,
                                "Calmar-Optimized Weight": calmar_weights
                            })
                            st.dataframe(weights_df)

                            # Calmar-optimized portfolio returns
                            calmar_rets = aligned_returns.dot(calmar_weights)
                            calmar_growth = pd.DataFrame({
                                'time': aligned_returns.index,
                                'Growth': (1 + calmar_rets).cumprod(),
                                'Label': 'Calmar-Optimized Portfolio'
                            })

                            # Current portfolio returns (using original weights)
                            current_rets = aligned_returns.dot(asset_weights)
                            current_growth = pd.DataFrame({
                                'time': aligned_returns.index,
                                'Growth': (1 + current_rets).cumprod(),
                                'Label': 'Current Portfolio'
                            })

                            # Combine data for plotting
                            combined_growth = pd.concat([current_growth, calmar_growth], ignore_index=True)

                            # Plot both lines using Altair
                            chart = alt.Chart(combined_growth).mark_line().encode(
                                x=alt.X('time:T', title='Time'),
                                y=alt.Y('Growth:Q', title='Cumulative Growth'),
                                color=alt.Color('Label:N', scale=alt.Scale(
                                    domain=['Calmar-Optimized Portfolio', 'Current Portfolio'],
                                    range=['Green', 'blue']  # Blue for current, purple for Calmar-optimized
                                )),
                                tooltip=[alt.Tooltip('time:T', title='Date', format='%Y-%m-%d'),
                                         alt.Tooltip('Label:N', title='Portfolio'),
                                         alt.Tooltip('Growth:Q', title='Growth', format='.4f')]
                            ).properties(
                                title="Current vs Calmar-Optimized Portfolio Growth",
                                width=600,
                                height=400
                            ).interactive()

                            st.altair_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Calmar optimization error: {e}")         

            with tab2:  # Individual Asset Metrics
                for asset in asset_names:
                    # Metrics Table Expander
                    with st.expander(f"{asset} - **Analysis**", expanded=False):
                        
                        if asset in asset_returns_dict:
                            # Display consolidated metrics table using existing function
                            display_individual_asset_periodic_metrics(
                                {asset: asset_returns_dict[asset]},
                                {asset: asset_benchmarks_dict[asset]},
                                risk_free_series
                            )

                    # Separate Expander for Growth Chart
                        st.subheader(f"{asset} - Growth Chart")
                        if asset in asset_returns_dict:
                            asset_returns = asset_returns_dict[asset].reindex(common_index, method='ffill').fillna(0.0)
                            bench_returns = asset_benchmarks_dict[asset].reindex(common_index, method='ffill').fillna(0.0)
                            asset_df = pd.DataFrame({
                                'time': common_index,
                                'Growth': (1 + asset_returns).cumprod(),
                                'Label': asset
                            })
                            bench_df = pd.DataFrame({
                                'time': common_index,
                                'Growth': (1 + bench_returns).cumprod(),
                                'Label': 'Benchmark'
                            })
                            combined_df = pd.concat([asset_df, bench_df], ignore_index=True)
                            chart = alt.Chart(combined_df).mark_line().encode(
                                x='time:T', y='Growth:Q', color='Label:N', tooltip=['time:T', 'Label:N', 'Growth:Q']
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)

                        st.subheader(f"{asset} - Drawdown Chart")
                        if asset in asset_returns_dict:
                            asset_cumret = (1 + asset_returns_dict[asset].reindex(common_index, method='ffill').fillna(0.0)).cumprod()
                            asset_drawdown = (asset_cumret / asset_cumret.cummax()) - 1
                            bench_cumret = (1 + asset_benchmarks_dict[asset].reindex(common_index, method='ffill').fillna(0.0)).cumprod()
                            bench_drawdown = (bench_cumret / bench_cumret.cummax()) - 1
                            drawdown_df = pd.DataFrame({
                                'time': np.tile(common_index, 2),
                                'Drawdown': pd.concat([asset_drawdown, bench_drawdown]).values,
                                'Label': [asset, 'Benchmark'] * len(common_index)
                            })
                            chart = alt.Chart(drawdown_df).mark_line().encode(
                                x='time:T', y='Drawdown:Q', color='Label:N', tooltip=['time:T', 'Label:N', 'Drawdown:Q']
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)

                        # Separate Expander for Dividend Decomposition
                        st.subheader(f"{asset} - Dividend Decomposition")
                        if asset in asset_returns_dict:
                            plot_dividend_decomposition(asset_data[asset_names.index(asset)], asset)

                        # Separate Expander for Return Distribution
                        st.subheader(f"{asset} - Return Distribution")
                        if asset in asset_returns_dict:
                            plot_return_distribution(asset_returns_dict[asset], asset)
                        
                                            
                        st.subheader(f"{asset} - Annual Returns Bar Plot")
                        if asset in asset_returns_dict:
                            df = asset_data[asset_names.index(asset)].copy()
                            if df.empty or 'total_return' not in df.columns:
                                st.warning(f"No valid return data for {asset}.")
                            else:
                                # Ensure the time column is datetime and create a Year column
                                df['time'] = pd.to_datetime(df['time'])
                                df['Year'] = df['time'].dt.year
                                # Group by year and compute the annual return
                                annual_returns = df.groupby('Year')['total_return'].apply(
                                    lambda x: np.prod(1 + x) - 1
                                ).reset_index().rename(columns={'total_return': 'Annual Return'})
                                
                                # Create a bar chart for this asset
                                bar_chart = alt.Chart(annual_returns).mark_bar().encode(
                                    x=alt.X('Year:O', title='Year'),
                                    y=alt.Y('Annual Return:Q', title='Annual Return', axis=alt.Axis(format='.2%')),
                                    color=alt.Color('Annual Return:Q', 
                                                    scale=alt.Scale(scheme='redyellowgreen', domainMid=0),
                                                    legend=None),
                                    tooltip=[
                                        alt.Tooltip('Year:O', title='Year'),
                                        alt.Tooltip('Annual Return:Q', format='.2%', title='Annual Return')
                                    ]
                                ).properties(
                                    title=f"Annual Returns for {asset}",
                                    width=500,
                                    height=300
                                ).interactive()
                                
                                st.altair_chart(bar_chart, use_container_width=True)

        except Exception as e:
            st.error(f"Portfolio calculation error: {str(e)}")
    else:
        st.info("Please upload data for all assets to view portfolio analytics.")

if __name__ == "__main__":
    main()
