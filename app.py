import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import skew, kurtosis
import streamlit as st
from typing import List, Dict, Tuple
import yfinance as yf
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

############################
#  Sidebar Configuration   #
############################

with st.sidebar:
    st.subheader("Benchmark Configuration")
    benchmark_ticker = st.text_input(
        "Benchmark Ticker (e.g., ^GSPC for S&P 500)",
        "^GSPC",
        help="Enter the ticker symbol for the benchmark."
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
    if len(asset_data) == 1:
        single_asset = asset_data[0]
        if single_asset.empty:
            st.warning("Asset data is empty. Drawdown chart cannot be displayed.")
            return
        cumret = (1 + single_asset['total_return']).cumprod()
        runmax = cumret.cummax()
        drawdown = (cumret / runmax) - 1
        tmp = pd.DataFrame({
            'time': single_asset['time'],
            'Drawdown': drawdown,
            'Label': asset_names[0]
        })
        chart_dfs.append(tmp)
        full_data = pd.concat(chart_dfs, ignore_index=True)
        full_data['time'] = pd.to_datetime(full_data['time'])
        if full_data.empty:
            st.error("No data for Drawdown chart. Check your inputs.")
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
    for i, df in enumerate(asset_data):
        if df.empty:
            continue
        cumret = (1 + df['total_return']).cumprod()
        runmax = cumret.cummax()
        drawdown = (cumret / runmax) - 1
        tmp = pd.DataFrame({
            'time': df['time'],
            'Drawdown': drawdown,
            'Label': asset_names[i]
        })
        chart_dfs.append(tmp)
    port_cumret = (1 + portfolio_returns).cumprod()
    port_runmax = port_cumret.cummax()
    port_drawdown = (port_cumret / port_runmax) - 1
    port_df = pd.DataFrame({
        'time': portfolio_returns.index,
        'Drawdown': port_drawdown.values,
        'Label': 'Portfolio'
    })
    chart_dfs.append(port_df)
    full_data = pd.concat(chart_dfs, ignore_index=True)
    full_data['time'] = pd.to_datetime(full_data['time'])
    if full_data.empty:
        st.error("No data for Drawdown chart. Check your inputs.")
        return
    selection = alt.selection_multi(fields=['Label'], bind='legend')
    color_scale = alt.Scale(
        domain=asset_names + ['Portfolio'],
        range=['blue', 'orange', 'green', 'purple', 'brown', 'steelblue', 'hotpink', 'gray', 'teal', 'indigo'][:len(asset_names)] + ['red']
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
    df.dropna(subset=['total_return', 'risk_free_rate'], inplace=True)
    if len(df) < 2:
        raise ValueError("Insufficient data points for analysis.")
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
def portfolio_metrics(asset_data: List[pd.DataFrame], weights: List[float], asset_names: List[str], benchmark_data: pd.DataFrame = None, risk_free_df: pd.DataFrame = None) -> Tuple[pd.Series, Dict[str, float], Dict[str, float], pd.DataFrame]:
    if not asset_data or not weights:
        raise ValueError("Asset data and weights cannot be empty.")
    if len(asset_data) != len(weights):
        raise ValueError("Number of assets must match number of weights.")
    if not np.isclose(sum(weights), 1.0, atol=1e-5):
        raise ValueError("Weights must sum to 1.0.")
    # For multi-asset case, use the separately uploaded risk-free rate if available.
    if risk_free_df is not None and not risk_free_df.empty:
        risk_free_rate = risk_free_df.set_index('time')['risk_free_rate']
    else:
        risk_free_rate = pd.concat([df[['time', 'risk_free_rate']] for df in asset_data if 'risk_free_rate' in df.columns]).groupby('time')['risk_free_rate'].mean()
    if len(asset_data) == 1:
        single_asset = asset_data[0]
        single_asset['time'] = pd.to_datetime(single_asset['time'], errors='coerce')
        single_asset = single_asset.sort_values('time').dropna(subset=['time'])
        if single_asset.empty:
            raise ValueError("Single asset data is empty or invalid.")
        asset_returns = single_asset[['time', 'total_return']].set_index('time')
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_data['time'] = pd.to_datetime(benchmark_data['time'], errors='coerce')
            benchmark_data = benchmark_data.set_index('time').sort_index()
            common_dates = asset_returns.index.intersection(benchmark_data.index)
            if common_dates.empty:
                raise ValueError("No overlapping dates between asset and benchmark.")
            aligned_asset_returns = asset_returns.loc[common_dates]
            aligned_benchmark_returns = benchmark_data['return'].loc[common_dates]
        else:
            st.warning("Benchmark data is empty. Metrics dependent on benchmark will not be calculated.")
            aligned_benchmark_returns = pd.Series([], dtype=float)
            aligned_asset_returns = asset_returns
        return _calculate_metrics(
            aligned_asset_returns['total_return'],
            aligned_asset_returns['total_return'] - single_asset['risk_free_rate'].reindex(aligned_asset_returns.index).fillna(0),
            single_asset['risk_free_rate'].reindex(aligned_asset_returns.index).fillna(0),
            aligned_benchmark_returns,
            None
        )
    aligned_returns = align_asset_data(asset_data=asset_data, asset_names=asset_names, benchmark_df=benchmark_data)
    if aligned_returns.empty:
        raise ValueError("No data after alignment (possibly no overlap).")
    risk_free_rate = risk_free_rate.reindex(aligned_returns.index).fillna(method='ffill')
    portfolio_returns = aligned_returns.dot(weights)
    excess_returns = portfolio_returns - risk_free_rate
    if benchmark_data is not None:
        benchmark_data['time'] = pd.to_datetime(benchmark_data['time'], errors='coerce')
        benchmark_data = benchmark_data.dropna(subset=['time'])
        benchmark_data = benchmark_data.set_index('time')
        common_dates = portfolio_returns.index.intersection(benchmark_data.index)
        portfolio_returns_aligned = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_data['return'].loc[common_dates]
        
    else:
        benchmark_returns = pd.Series([], dtype=float)
    try:
        risk_decomp_table = compute_risk_contributions(aligned_returns.loc[common_dates], weights)
    except Exception as e:
        st.error(f"Error computing risk decomposition: {e}")
        risk_decomp_table = pd.DataFrame()
    
    return _calculate_metrics(
        portfolio_returns_aligned,
        excess_returns.loc[common_dates],
        risk_free_rate.loc[common_dates],
        benchmark_returns,
        risk_decomp_table
    )

def _calculate_metrics(portfolio_returns, excess_returns, risk_free_rate, benchmark_returns, risk_decomp_table: pd.DataFrame = None):
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
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    beta = covariance / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
    risk_free_rate_annual = (1 + risk_free_rate.mean()) ** 12 - 1
    mean_benchmark_return_monthly = benchmark_returns.mean()
    mean_benchmark_return_annual = mean_benchmark_return_monthly * 12
    alpha = mean_return_annual - (risk_free_rate_annual + beta * (mean_benchmark_return_annual - risk_free_rate_annual))
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    aligned_portfolio_returns = portfolio_returns.loc[common_dates]
    aligned_benchmark_returns = benchmark_returns.loc[common_dates]
    tracking_error = (aligned_portfolio_returns - aligned_benchmark_returns).std()
    mean_return_monthly = aligned_portfolio_returns.mean()
    mean_benchmark_return = aligned_benchmark_returns.mean()
    information_ratio = (mean_return_monthly - mean_benchmark_return) / tracking_error if tracking_error > 0 else 0
    benchmark_up = benchmark_returns[benchmark_returns > 0]
    portfolio_up = portfolio_returns.loc[benchmark_up.index.intersection(portfolio_returns.index)]
    benchmark_down = benchmark_returns[benchmark_returns < 0]
    portfolio_down = portfolio_returns.loc[benchmark_down.index.intersection(portfolio_returns.index)]
    if not benchmark_up.empty:
        portfolio_up = portfolio_returns.loc[benchmark_up.index]
        upside_capture = (portfolio_up.mean() / benchmark_up.mean() if benchmark_up.mean() > 0 else 0)
    else:
        upside_capture = 0
    if not benchmark_down.empty:
        portfolio_down = portfolio_returns.loc[benchmark_down.index]
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
                                               bench_returns: pd.Series,
                                               risk_free_series: pd.Series,
                                               periods: List[str] = ["1Y", "3Y", "5Y", "ITD"]) -> None:
    """
    For each asset (from asset_returns_dict), computes periodic metrics using the common benchmark and risk‐free series.
    Builds and displays a MultiIndex DataFrame with rows as metric names and columns as (Asset, Period).
    """
    groups = {}
    for asset, returns_series in asset_returns_dict.items():
        asset_metrics = {}
        for period in periods:
            if period == "ITD":
                metrics = calc_periodic_metrics(returns_series, bench_returns, risk_free_series)
            else:
                # Assume period is like "1Y", "3Y", "5Y" – convert to months.
                months = int(period[:-1]) * 12
                end_date = returns_series.index.max()
                start_date = end_date - pd.DateOffset(months=months)
                metrics = calc_periodic_metrics(returns_series.loc[start_date:end_date],
                                                bench_returns.loc[start_date:end_date],
                                                risk_free_series.loc[start_date:end_date])
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


    formatted_table = df_table.apply(format_row, axis=1)
    st.table(formatted_table)


#########################
#  The Main Function    #
#########################
def main():
    st.title("Portfolio Analytics Dashboard")
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")
        num_assets = st.number_input("**Number of Assets**", min_value=1, max_value=10, value=1, help="Select the number of assets in your portfolio.")
        st.markdown("---")
        dividend_as_yield = st.radio(
            "**Dividend Input Type**", 
            ("Yield", "Actual Amount"), 
            index=0, 
            help="Specify whether dividends are provided as a yield (%) or actual amounts.") == "Yield"
        st.markdown("---")
    asset_data = []
    asset_weights = []
    asset_names = []
    total_weight = 0
    for i in range(num_assets):
        st.sidebar.subheader(f"**Asset {i+1} Configuration**")
        asset_name = st.sidebar.text_input(f"Asset {i+1} Name", f"Asset {i+1}", key=f"name_{i}", help="Provide a name for the asset.")
        uploaded_file = st.sidebar.file_uploader(
            f"Upload CSV for {asset_name}", 
            type=["csv"], 
            key=f"file_{i}", 
            help="Upload a CSV file containing the asset's data."
        )
        weight_percent = st.sidebar.slider(
            f"Weight (%) for {asset_name}", 
            min_value=0, 
            max_value=100, 
            value=int(100 / num_assets), 
            step=1, 
            key=f"weight_{i}", 
            help="Set the portfolio allocation weight for this asset."
        )
        total_weight += weight_percent
        w_frac = weight_percent / 100
        st.sidebar.markdown("---")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df = adjust_column_names(df)
                if (('price' not in df.columns) or (df.get('price').isnull().all())) and ('return' in df.columns):
                    df['price'] = (1 + df['return']).cumprod()
                elif 'price' in df.columns:
                    if 'return' not in df.columns:
                        df['return'] = df['price'].pct_change()
                validate_columns(df)
                df['time'] = pd.to_datetime(df['time'])
                df['time'] = df['time'].dt.to_period('M').dt.to_timestamp('M')
                # **NEW: Merge risk-free rate data into the asset data**
                if not risk_free_df.empty:
                    df = pd.merge(df, risk_free_df, on='time', how='left')
                df, raw_metrics, annual_metrics = calculate_metrics(df, dividend_as_yield)
                asset_data.append(df)
                asset_weights.append(w_frac)
                asset_names.append(asset_name)
            except Exception as e:
                st.error(f"Error processing {asset_name}: {str(e)}")
                asset_data.append(pd.DataFrame())
                asset_weights.append(w_frac)
                asset_names.append(asset_name)
        else:
            st.warning(f"No file uploaded for {asset_name}. Adding empty data for this asset.")
            asset_data.append(pd.DataFrame())
            asset_weights.append(w_frac)
            asset_names.append(asset_name)
    if not np.isclose(total_weight, 100, atol=1e-5):
        st.sidebar.error("Error: The total weight of all assets must equal 100%.")
        return
    asset_weights = normalize_weights(asset_weights)
    if benchmark_ticker and all(not df.empty for df in asset_data):
        try:
            start_date, end_date = validate_time_ranges(asset_data)
            benchmark_data = fetch_benchmark_data(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except ValueError as e:
            st.error(f"Time range validation failed: {e}")
            return
    else:
        benchmark_data = pd.DataFrame()
    if all(not df.empty for df in asset_data):
        try:
            start_date, end_date = validate_time_ranges(asset_data)
            benchmark_data = fetch_benchmark_data(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not benchmark_data.empty:
                st.warning("Benchmark data fetched...")
            else:
                st.warning("Benchmark data is empty after fetching. Please check the ticker and date range.")
            total_weight = sum(asset_weights)
            if total_weight > 0:
                asset_weights = [w / total_weight for w in asset_weights]
            portfolio_rets, raw_port_metrics, annual_port_metrics, risk_decomp_table = portfolio_metrics(
                asset_data=asset_data,
                weights=asset_weights,
                asset_names=asset_names,
                benchmark_data=(benchmark_data if 'benchmark_data' in locals() and not benchmark_data.empty else None),
                risk_free_df = risk_free_df if not risk_free_df.empty else None
            )
            st.header("Portfolio Analysis")
            st.markdown("---")
            metrics_tab, risk_tab = st.tabs(["Portfolio Metrics", "Risk Decomposition"])
            with metrics_tab:
                st.header("Periodic Annualized Metrics")
                # Compute risk-free series from your risk_free_df (assumes monthly frequency)
                if not risk_free_df.empty:
                    risk_free_series = risk_free_df.set_index('time')['risk_free_rate']
                    # If necessary, reindex risk_free_series to match portfolio returns frequency:
                    risk_free_series = risk_free_series.reindex(portfolio_rets.index, method='ffill')
                else:
                    st.error("Risk-free rate data is required for periodic metrics.")
                    risk_free_series = pd.Series(dtype=float)

                # In portfolio_metrics you already computed benchmark_returns.
                # Ensure benchmark_returns is available; if not, re-create it from benchmark_data:
                if not benchmark_data.empty:
                    benchmark_returns = benchmark_data.set_index('time')['return']
                    benchmark_returns = benchmark_returns.reindex(portfolio_rets.index, method='ffill')
                else:
                    st.error("Benchmark data is required for periodic metrics.")
                    benchmark_returns = pd.Series(dtype=float)
            # Now call the new display function with fund (portfolio) returns, benchmark returns, and risk_free_series:
            display_full_periodic_table(portfolio_rets, benchmark_returns, risk_free_series)
            
            with risk_tab:
                st.subheader("Portfolio Risk Decomposition")
                risk_decomp_df = pd.DataFrame()
                if 'risk_decomp_table' in locals():
                    if not risk_decomp_table.empty:
                        risk_decomp_df = risk_decomp_table.reset_index().rename(columns={'index': 'Asset'})
                        st.dataframe(risk_decomp_df.round(4))
                        st.write("Risk decomposition shows the contribution of each asset to portfolio volatility.")
                    else:
                        st.warning("Risk decomposition could not be calculated.")
                    if not risk_decomp_df.empty:
                        st.subheader("Risk Decomposition Pie Chart")
                        pie_chart = alt.Chart(risk_decomp_df).mark_arc().encode(
                            theta=alt.Theta(field='Percent of Risk', type='quantitative'),
                            color=alt.Color(field='Asset', type='nominal', legend=alt.Legend(title="Assets")),
                            tooltip=['Asset', alt.Tooltip('Percent of Risk', format=".2f")]
                        ).properties(title="Contribution to Portfolio Risk")
                        st.altair_chart(pie_chart, use_container_width=True)
                else:
                    st.warning("No risk decomposition data available.")
            tab1, tab2, tab3 = st.tabs(["Growth Charts", "Portfolio Growth", "Drawdown"])
            with tab1:
                st.subheader("Growth of Each Asset vs. Benchmark")
                chart_dfs = []
                for i, df in enumerate(asset_data):
                    if df.empty:
                        continue
                    tmp = df[['time']].copy()
                    tmp['Growth'] = (1 + df['total_return']).cumprod()
                    tmp['Label'] = asset_names[i]
                    chart_dfs.append(tmp)
                if not portfolio_rets.empty:
                    port_tmp = pd.DataFrame({'time': portfolio_rets.index})
                    port_tmp['Growth'] = (1 + portfolio_rets).cumprod()
                    port_tmp['Label'] = 'Portfolio'
                    chart_dfs.append(port_tmp)
                if 'benchmark_data' in locals() and not benchmark_data.empty:
                    benchmark_tmp = benchmark_data[['time']].copy()
                    benchmark_tmp['Growth'] = (1 + benchmark_data['return']).cumprod()
                    benchmark_tmp['Label'] = 'Benchmark'
                    chart_dfs.append(benchmark_tmp)
                if chart_dfs:
                    full_data = pd.concat(chart_dfs, ignore_index=True)
                    full_data['time'] = pd.to_datetime(full_data['time'])
                else:
                    st.warning("No data available to plot.")
                    return
                filtered_data = full_data[full_data['Label'] != 'Portfolio']
                color_scale = alt.Scale(
                    domain=filtered_data['Label'].unique().tolist(),
                    range=['blue', 'orange', 'green', 'purple', 'brown', 'red', 'pink', 'teal', 'gray'][:len(filtered_data['Label'].unique())]
                )
                selection = alt.selection_multi(fields=['Label'], bind='legend')
                line_chart = (
                    alt.Chart(filtered_data)
                    .mark_line()
                    .encode(
                        x=alt.X('time:T', title='Time'),
                        y=alt.Y('Growth:Q', title='Cumulative Growth'),
                        color=alt.Color('Label:N', scale=color_scale, legend=alt.Legend(title="Legend")),
                        tooltip=['time:T', 'Label:N', 'Growth:Q']
                    )
                    .add_selection(selection)
                    .transform_filter(selection)
                    .interactive()
                )
                st.altair_chart(line_chart, use_container_width=True)
            with tab2:
                st.subheader("Portfolio Growth")
                if len(asset_data) > 1:
                    portfolio_growth_df = pd.DataFrame({
                        'time': portfolio_rets.index,
                        'Growth': (1 + portfolio_rets).cumprod(),
                        'Label': 'Portfolio'
                    })
                    portfolio_growth_df['time'] = pd.to_datetime(portfolio_growth_df['time'], errors='coerce')
                    portfolio_growth_df = portfolio_growth_df.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time')
                    if 'benchmark_data' in locals() and not benchmark_data.empty:
                        benchmark_growth_df = pd.DataFrame({
                            'time': pd.to_datetime(benchmark_data['time'], errors='coerce'),
                            'Growth': (1 + benchmark_data['return']).cumprod(),
                            'Label': 'Benchmark'
                        })
                        benchmark_growth_df = benchmark_growth_df.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time')
                        combined_growth_df = pd.concat([portfolio_growth_df, benchmark_growth_df], ignore_index=True)
                        st.subheader("Cumulative Portfolio Growth Data")
                        st.dataframe(portfolio_growth_df)
                        growth_chart = alt.Chart(combined_growth_df).mark_line().encode(
                            x=alt.X('time:T', title='Time'),
                            y=alt.Y('Growth:Q', title='Cumulative Growth'),
                            color=alt.Color('Label:N', legend=alt.Legend(title="Legend")),
                            tooltip=['time:T', 'Label:N', 'Growth:Q']
                        ).properties(title="Portfolio vs Benchmark Growth").interactive()
                        st.altair_chart(growth_chart, use_container_width=True)
                    else:
                        st.warning("Benchmark data is empty. Unable to plot benchmark growth.")
                else:
                    st.info("Portfolio growth graph is not displayed for a single asset.")
            with tab3:
                st.subheader("Drawdown Over Time (All Assets + Portfolio)")
                plot_drawdown_of_assets(asset_data, asset_names, portfolio_rets)
            st.header("Yearly/Monthly Performance Table")
            try:
                if not portfolio_rets.empty and 'benchmark_data' in locals() and not benchmark_data.empty:
                    portfolio_rets.index = pd.to_datetime(portfolio_rets.index)
                    portfolio_returns_yearly = portfolio_rets.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                    benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])
                    benchmark_yearly = (
                        benchmark_data.set_index('time')['return']
                        .resample('Y')
                        .apply(lambda x: (1 + x).prod() - 1)
                    )
                    portfolio_returns_yearly.index = portfolio_returns_yearly.index.year
                    benchmark_yearly.index = benchmark_yearly.index.year
                    portfolio_monthly = portfolio_rets.reset_index()
                    portfolio_monthly.columns = ['time', 'return']
                    portfolio_monthly['Year'] = portfolio_monthly['time'].dt.year
                    portfolio_monthly['Month'] = portfolio_monthly['time'].dt.month
                    monthly_table = portfolio_monthly.pivot(index='Year', columns='Month', values='return')
                    all_months = pd.MultiIndex.from_product([monthly_table.index, range(1, 13)], names=['Year', 'Month'])
                    monthly_table = (portfolio_monthly.set_index(['Year', 'Month']).reindex(all_months)['return'].unstack())
                    monthly_table['Total'] = portfolio_returns_yearly.reindex(monthly_table.index, fill_value=np.nan)
                    monthly_table['Benchmark'] = benchmark_yearly.reindex(monthly_table.index, fill_value=np.nan)
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month_mapping = {i + 1: month_names[i] for i in range(12)}
                    monthly_table.rename(columns=month_mapping, inplace=True)
                    monthly_table.index = monthly_table.index.map(str)
                    st.write("Monthly Table:")
                    st.write("""
                        This table summarizes portfolio performance on a monthly and yearly basis.
                        It includes:
                        - **Monthly Portfolio Returns**: January to December.
                        - **Yearly Total Portfolio Return**.
                        - **Yearly Benchmark Return**.
                    """)
                    st.dataframe(monthly_table.style.format("{:.2%}"))
                    csv_data = monthly_table.to_csv(index=True)
                    st.download_button(
                        label="Download Yearly/Monthly Performance Table as CSV",
                        data=csv_data,
                        file_name="yearly_monthly_performance_table.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Portfolio returns or benchmark data is not available.")
            except Exception as e:
                st.error(f"Error generating the table: {e}")
            
            
            # --- Build dictionary for individual asset returns ---
            asset_returns_dict = {}
            for name, df in zip(asset_names, asset_data):
                if not df.empty:
                    asset_returns_dict[name] = df.set_index('time')['total_return']

            # --- Ensure benchmark_returns and risk_free_series are aligned ---
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data.set_index('time')['return']
                benchmark_returns = benchmark_returns.reindex(portfolio_rets.index, method='ffill')
            else:
                st.error("Benchmark data is required for periodic metrics.")
                benchmark_returns = pd.Series(dtype=float)

            if not risk_free_df.empty:
                risk_free_series = risk_free_df.set_index('time')['risk_free_rate']
                risk_free_series = risk_free_series.reindex(portfolio_rets.index, method='ffill')
            else:
                st.error("Risk-free rate data is required for periodic metrics.")
                risk_free_series = pd.Series(dtype=float)

            # --- Reindex to a common index for safety ---
            common_index = portfolio_rets.index.intersection(benchmark_returns.index).intersection(risk_free_series.index)
            portfolio_rets = portfolio_rets.reindex(common_index)
            benchmark_returns = benchmark_returns.reindex(common_index)
            risk_free_series = risk_free_series.reindex(common_index, method='ffill')

            # --- Reindex each individual asset's return series to the common index ---
            for asset in asset_returns_dict:
                asset_returns_dict[asset] = asset_returns_dict[asset].reindex(common_index)

            # --- Call the new function for individual asset periodic metrics ---
            display_individual_asset_periodic_metrics(asset_returns_dict, benchmark_returns, risk_free_series)



        except Exception as e:
            st.error(f"Portfolio calculation error: {str(e)}")
    else:
        st.info("Please upload data for all assets to view portfolio analytics.")

if __name__ == "__main__":
    main()
