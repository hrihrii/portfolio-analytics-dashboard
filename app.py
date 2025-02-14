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

def build_hierarchical_table(master_df):
    """
    Create a hierarchical master table with aggregate rows.
    Returns a multi-index DataFrame with aggregated totals at Asset and Sub-Asset levels.
    """
    rows = []
    for asset in master_df["Asset Type"].unique():
        asset_df = master_df[master_df["Asset Type"] == asset]
        asset_total = asset_df["Capital Allocation (Millions)"].sum()
        if asset_total > 0:
            asset_weighted_return = (asset_df["Capital Allocation (Millions)"] * asset_df["Expected Return (%)"]).sum() / asset_total
            asset_weighted_vol = (asset_df["Capital Allocation (Millions)"] * asset_df["Hypothetical Volatility (%)"]).sum() / asset_total
        else:
            asset_weighted_return, asset_weighted_vol = 0, 0
        # Asset total row
        rows.append({
            "Asset Type": asset,
            "Sub-Asset Type": "",
            "Fund": "TOTAL",
            "Capital Allocation (Millions)": asset_total,
            "Weighted Expected Return (%)": asset_weighted_return,
            "Weighted Volatility (%)": asset_weighted_vol,
            "Fund Weight (%)": np.nan
        })
        for sub in asset_df["Sub-Asset Type"].unique():
            sub_df = asset_df[asset_df["Sub-Asset Type"] == sub]
            sub_total = sub_df["Capital Allocation (Millions)"].sum()
            if sub_total > 0:
                sub_weighted_return = (sub_df["Capital Allocation (Millions)"] * sub_df["Expected Return (%)"]).sum() / sub_total
                sub_weighted_vol = (sub_df["Capital Allocation (Millions)"] * sub_df["Hypothetical Volatility (%)"]).sum() / sub_total
            else:
                sub_weighted_return, sub_weighted_vol = 0, 0
            # Sub-Asset total row
            rows.append({
                "Asset Type": asset,
                "Sub-Asset Type": sub,
                "Fund": "TOTAL",
                "Capital Allocation (Millions)": sub_total,
                "Weighted Expected Return (%)": sub_weighted_return,
                "Weighted Volatility (%)": sub_weighted_vol,
                "Fund Weight (%)": np.nan
            })
            # Individual funds
            for _, fund_row in sub_df.iterrows():
                rows.append({
                    "Asset Type": asset,
                    "Sub-Asset Type": sub,
                    "Fund": fund_row["Fund"],
                    "Capital Allocation (Millions)": fund_row["Capital Allocation (Millions)"],
                    "Weighted Expected Return (%)": fund_row["Expected Return (%)"],
                    "Weighted Volatility (%)": fund_row["Hypothetical Volatility (%)"],
                    "Fund Weight (%)": fund_row["Fund Weight (%)"]
                })
    hierarchical_df = pd.DataFrame(rows)
    hierarchical_df.set_index(["Asset Type", "Sub-Asset Type", "Fund"], inplace=True)
    return hierarchical_df

def style_master_table(df):
    """
    Apply styling to the hierarchical master table:
      - Merged cell effect simulated via multi-index rows.
      - Background colors: Light Blue for Asset Totals, Light Gray for Sub-Asset Totals, White for funds.
      - Totals rows are bold.
      - Conditional formatting: High volatility (>15%) and low returns (<5%).
      - Increased row height and centered text.
    """
    def row_style(row):
        asset, sub, fund = row.name
        if fund == "TOTAL":
            if sub == "":
                return "background-color: #ADD8E6; font-weight: bold;"  # Light Blue for Asset total
            else:
                return "background-color: #D3D3D3; font-weight: bold;"  # Light Gray for Sub-Asset total
        return "background-color: white;"
    
    styled = df.style.apply(lambda row: [row_style(row)] * len(row), axis=1)
    
    def format_vol(val):
        try:
            if float(val) > 15:
                return "background-color: #FFCCCC;"  # light red
        except:
            return ""
        return ""
    
    def format_return(val):
        try:
            if float(val) < 5:
                return "background-color: #EEEEEE;"  # light gray
        except:
            return ""
        return ""
    
    styled = styled.applymap(format_vol, subset=["Weighted Volatility (%)"])
    styled = styled.applymap(format_return, subset=["Weighted Expected Return (%)"])
    
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('padding', '10px'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('padding', '10px'), ('text-align', 'center')]},
        {'selector': 'tr', 'props': [('height', '50px')]}
    ])
    return styled

def plot_enhanced_sunburst(master_df):
    """
    Create an enhanced, interactive sunburst chart with:
      - Drill-down interactivity
      - Detailed hover information
      - Smooth transitions
      - Hierarchical color gradient (using the Asset Type for color)
    """
    fig = px.sunburst(
        master_df,
        path=["Asset Type", "Sub-Asset Type", "Fund"],
        values="Capital Allocation (Millions)",
        color="Asset Type",
        hover_data={
            "Capital Allocation (Millions)": True,
            "Expected Return (%)": True,
            "Hypothetical Volatility (%)": True,
            "Fund Weight (%)": True
        },
        branchvalues="total",
        title="Enhanced Portfolio Allocation Sunburst"
    )
    # Update layout for smooth transitions and improved interactivity
    fig.update_traces(textinfo="label+percent entry", hovertemplate="<b>%{label}</b><br>Allocation: %{value:.2f}M<br>Percent: %{percentParent:.1%}<extra></extra>")
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        transition=dict(duration=500),
        uniformtext=dict(minsize=12, mode='hide')
    )
    st.plotly_chart(fig, use_container_width=True)

############################################
# Page 2: Portfolio Structuring Tool (Using AgGrid)
############################################
def page2_portfolio_structuring():
    st.title("Investment Portfolio Structuring Tool")
    st.markdown("""
    **Step-by-Step Guide:**
    
    1. **Define Asset Classes**  
    2. **Define Sub-Asset Classes** for each asset  
    3. **Enter Fund-Level Data** for each sub-asset (fund name, capital allocation in millions, expected return, volatility)  
    4. The tool computes aggregates at every level and displays a **hierarchical master table** and an enhanced interactive **sunburst chart**.
    """)
    
    # Step 1: Define Asset Classes
    st.markdown("### Step 1: Define Asset Classes")
    num_assets = st.number_input("Number of asset classes:", min_value=1, max_value=10, value=1, step=1, key="num_assets")
    asset_classes = [st.text_input(f"Asset Class #{i+1} Name", key=f"asset_class_{i}") for i in range(int(num_assets))]
    
    # Step 2: Define Sub-Asset Classes
    st.markdown("### Step 2: Define Sub-Asset Classes")
    sub_assets = {}
    for asset in asset_classes:
        if asset.strip() == "":
            continue
        with st.expander(f"Sub-Asset Classes for **{asset}**", expanded=True):
            num_sub = st.number_input(f"Number of sub-asset classes for **{asset}**:", min_value=1, max_value=10, value=1, step=1, key=f"num_sub_{asset}")
            sub_list = [st.text_input(f"Sub-Asset Class #{j+1} for **{asset}**", key=f"{asset}_sub_{j}") for j in range(int(num_sub))]
            sub_assets[asset] = sub_list

    # Step 3: Define Funds for Each Sub-Asset using AgGrid
    st.markdown("### Step 3: Define Funds for Each Sub-Asset")
    funds_data = []
    for asset in asset_classes:
        if asset.strip() == "":
            continue
        for sub in sub_assets.get(asset, []):
            if sub.strip() == "":
                continue
            with st.expander(f"Funds for **{asset} → {sub}**", expanded=True):
                num_funds = st.number_input(f"Number of funds in **{asset} → {sub}**:", min_value=1, max_value=20, value=1, step=1, key=f"num_funds_{asset}_{sub}")
                default_data = {
                    "Fund": [f"Fund {i+1}" for i in range(int(num_funds))],
                    "Capital Allocation (Millions)": [0] * int(num_funds),
                    "Expected Return (%)": [0] * int(num_funds),
                    "Hypothetical Volatility (%)": [0] * int(num_funds)
                }
                df_funds = pd.DataFrame(default_data)
                df_funds["Asset Type"] = asset
                df_funds["Sub-Asset Type"] = sub

                gb = GridOptionsBuilder.from_dataframe(df_funds)
                gb.configure_default_column(editable=True, filter=True, sortable=True)
                grid_options = gb.build()

                grid_response = AgGrid(
                    df_funds,
                    gridOptions=grid_options,
                    editable=True,
                    height=200,
                    fit_columns_on_grid_load=True,
                    key=f"aggrid_{asset}_{sub}"
                )
                edited_funds = pd.DataFrame(grid_response["data"])
                funds_data.append(edited_funds)
    
    # Step 4: Build Master Table and Visualization
    st.markdown("## Step 4: Master Portfolio Table & Enhanced Sunburst Chart")
    if funds_data:
        master_df = pd.concat(funds_data, ignore_index=True)
        total_capital = master_df["Capital Allocation (Millions)"].sum()
        if total_capital > 0:
            master_df["Fund Weight (%)"] = master_df["Capital Allocation (Millions)"] / total_capital * 100
        else:
            master_df["Fund Weight (%)"] = 0

        hierarchical_df = build_hierarchical_table(master_df)
        st.markdown("### Hierarchical Master Portfolio Table")
        st.dataframe(style_master_table(hierarchical_df))
        
        st.markdown("### Enhanced Interactive Sunburst Chart")
        plot_enhanced_sunburst(master_df)
        
        csv_data = master_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Master Portfolio Data as CSV",
            data=csv_data,
            file_name="portfolio_master_table.csv",
            mime="text/csv"
        )
    else:
        st.info("No fund data provided yet. Please add funds in Step 3.")

def style_master_table(df):
    """
    Enhances the hierarchical master table:
      - Merges cells visually for Asset Types and Sub-Asset Types.
      - Color codes total rows distinctly.
      - Adds conditional formatting for volatility and returns.
      - Aligns numbers properly for clarity.
    """

    def row_style(row):
        asset, sub, fund = row.name

        if fund == "TOTAL":
            if sub == "":
                return ["background-color: #A1D6E2; font-weight: bold; text-align: center;"] * len(row)  # Light Blue for Asset Totals
            else:
                return ["background-color: #D3D3D3; font-weight: bold; text-align: center;"] * len(row)  # Light Gray for Sub-Asset Totals
        
        return ["background-color: white; text-align: center;"] * len(row)

    def format_vol(val):
        try:
            if float(val) > 15:
                return "background-color: #FFB6C1; color: black; font-weight: bold;"  # Light Red for High Volatility
        except:
            return ""
        return ""

    def format_return(val):
        try:
            if float(val) < 5:
                return "background-color: #E0E0E0; color: black;"  # Light Gray for Low Returns
        except:
            return ""
        return ""

    styled = df.style.apply(lambda row: row_style(row), axis=1)
    styled = styled.applymap(format_vol, subset=["Weighted Volatility (%)"])
    styled = styled.applymap(format_return, subset=["Weighted Expected Return (%)"])

    # Apply a uniform table style
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('padding', '10px'), ('text-align', 'center'), ('background-color', '#4F81BD'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('padding', '10px'), ('text-align', 'center')]},
        {'selector': 'tr', 'props': [('height', '40px')]}
    ])
    
    return styled


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
        risk_free_list = [df[['time', 'risk_free_rate']] for df in asset_data if 'risk_free_rate' in df.columns]
        if risk_free_list:
            risk_free_rate = pd.concat(risk_free_list).groupby('time')['risk_free_rate'].mean()
        else:
            # If no risk‑free rate is available, assume 0% for all dates.
            aligned = align_asset_data(asset_data, asset_names)
            risk_free_rate = pd.Series(0.0, index=aligned.index)

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
    csv_data = formatted_df.to_csv(index=True)
    st.download_button(
        label="Download Individual Asset Periodic Metrics CSV",
        data=csv_data,
        file_name="individual_asset_periodic_metrics.csv",
        mime="text/csv"
        )
    with st.expander("Show Data Window Details for Each Asset"):
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


    formatted_table = df_table.apply(format_row, axis=1)
    st.table(formatted_table)
    
    # --- Add a disclaimer with the date up to which the fund returns (and thus the metrics) are calculated ---
    one_year_end = fund_returns.index.max()
    one_year_start = one_year_end - pd.DateOffset(months=12)
    three_year_start = one_year_end - pd.DateOffset(months=36)
    five_year_start = one_year_end - pd.DateOffset(months=60)
    
    csv_data = df_table.to_csv(index=True)
    st.download_button(
        label="Download Periodic Annualized Stats CSV",
        data=csv_data,
        file_name="periodic_annualized_stats.csv",
        mime="text/csv"
    )
    
    st.caption("1Y metrics are based on data from {} to {}"
               .format(one_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    st.caption("3Y metrics are based on data from {} to {}"
               .format(three_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    st.caption("5Y metrics are based on data from {} to {}"
               .format(five_year_start.strftime("%Y-%m-%d"), one_year_end.strftime("%Y-%m-%d")))
    
    

    
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
        "Final Weight (%)": (final_weights * 100).round(2)
    })
    
    st.subheader("Final Portfolio Weights")
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
    Computes the annual return for each asset per year and displays a grouped bar chart.
    
    For each asset, the annual return for a given year is calculated as:
        Annual Return = (∏ (1 + monthly return)) - 1
    """
    rows = []
    for asset, df in zip(asset_names, asset_data):
        if df.empty or 'total_return' not in df.columns:
            continue
        # Ensure the time column is datetime and create a Year column.
        df['time'] = pd.to_datetime(df['time'])
        df['Year'] = df['time'].dt.year
        # Group by year and compute the annual return.
        grouped = df.groupby('Year')['total_return'].apply(
            lambda x: np.prod(1 + x) - 1
        ).reset_index().rename(columns={'total_return': 'Annual Return'})
        grouped['Asset'] = asset
        rows.append(grouped)
    if not rows:
        st.warning("No valid asset data available to compute annual returns.")
        return
    data = pd.concat(rows, ignore_index=True)
    
    # Create a grouped bar chart using Altair.
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        xOffset=alt.XOffset('Asset:N'),
        y=alt.Y('Annual Return:Q', title='Annual Return', axis=alt.Axis(format='.2%')),
        color=alt.Color('Asset:N', title='Asset'),
        tooltip=[
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('Asset:N', title='Asset'),
            alt.Tooltip('Annual Return:Q', format='.2%', title='Annual Return')
        ]
    ).properties(
        title="Annual Returns by Asset Over Time"
    )
    st.altair_chart(chart, use_container_width=True)


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
#########################
def main():
    st.title("Portfolio Analytics Dashboard")
    
    page = st.sidebar.radio(
        "Select Page",
        options=["Portfolio Analytics", "Page 2: Additional Analysis"],
        index=0,
        help="Choose which page to display."
    )
    if page == "Portfolio Analytics":
        # === Sidebar: Global Configuration & Asset Upload ===
        with st.sidebar:
            st.header("Configuration")
            num_assets = st.number_input(
                "**Number of Assets**", 
                min_value=1, 
                max_value=20, 
                value=1, 
                help="Select the number of assets in your portfolio."
            )
            st.markdown("---")
            
            # Bring back the dividend type toggle.
            dividend_as_yield = st.radio(
                "**Dividend Input Type**", 
                ("Yield", "Actual Amount"), 
                index=0, 
                help="Specify whether dividends are provided as a yield (%) or actual amounts."
            ) == "Yield"
            st.markdown("---")
            
            # --- Asset Upload Section ---
            st.header("Asset Upload")
            asset_data = []
            asset_names = []
            for i in range(num_assets):
                st.subheader(f"Asset {i+1}")
                asset_name = st.text_input(
                    f"Asset {i+1} Name", 
                    f"Asset {i+1}", 
                    key=f"name_{i}", 
                    help="Provide a name for the asset."
                )
                uploaded_file = st.file_uploader(
                    f"Upload CSV for {asset_name}", 
                    type=["csv"], 
                    key=f"file_{i}", 
                    help="Upload a CSV file containing the asset's data."
                )
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        df = adjust_column_names(df)
                        # If 'price' is missing but 'return' is available, compute price series
                        if (('price' not in df.columns) or (df.get('price').isnull().all())) and ('return' in df.columns):
                            df['price'] = (1 + df['return']).cumprod()
                        elif 'price' in df.columns and ('return' not in df.columns):
                            df['return'] = df['price'].pct_change()
                        validate_columns(df)
                        df['time'] = pd.to_datetime(df['time'])
                        df['time'] = df['time'].dt.to_period('M').dt.to_timestamp('M')
                        # Merge risk‑free rate data if available
                        if not risk_free_df.empty:
                            df = pd.merge(df, risk_free_df, on='time', how='left')
                        df, raw_metrics, annual_metrics = calculate_metrics(df, dividend_as_yield)
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
            
            st.markdown("---")
            # --- Portfolio Configuration Section ---
            st.header("Portfolio Configuration")
            asset_allocations = []
            total_allocation = 0.0
            for asset in asset_names:
                allocation = st.number_input(
                    f"Allocation (%) for {asset}", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=round(100/num_assets, 2), 
                    step=0.01, 
                    format="%.2f", 
                    key=f"alloc_{asset}"
                )
                asset_allocations.append(allocation)
                total_allocation += allocation
            st.write(f"**Total Allocation:** {total_allocation:.2f}%")
            if not np.isclose(total_allocation, 100.0, atol=0.01):
                st.error("Error: The total allocation of all assets must equal 100%.")
                st.stop()  # Stop further processing if the allocations are not correct
            
            # Normalize the allocations to create asset weights (if desired)
            asset_weights = normalize_weights(asset_allocations)
        
        # === End of Sidebar; continue processing on the main page ===
        # Fetch benchmark data if a ticker is provided and asset data is complete.
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
                benchmark_data = pd.read_csv(benchmark_file)
                benchmark_data = adjust_column_names(benchmark_data)
                # If price exists but return is missing, compute the return.
                if ('price' in benchmark_data.columns) and ('return' not in benchmark_data.columns):
                    benchmark_data['return'] = benchmark_data['price'].pct_change()
                benchmark_data['time'] = pd.to_datetime(benchmark_data['time'])
                benchmark_data['time'] = benchmark_data['time'].dt.to_period('M').dt.to_timestamp('M')
            else:
                benchmark_data = pd.DataFrame()

        
        if all(not df.empty for df in asset_data):
            try:
                # Revalidate time ranges and re-fetch benchmark data if necessary.
                start_date, end_date = validate_time_ranges(asset_data)
                # Only re-fetch benchmark data if the user chose "Fetch from Ticker".
                if benchmark_source == "Fetch from Ticker":
                    benchmark_data = fetch_benchmark_data(benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    if not benchmark_data.empty:
                        st.warning("Benchmark data fetched...")
                    else:
                        st.warning("Benchmark data is empty after fetching. Please check the ticker and date range.")
                    
                # (Optional) Normalize asset_weights again.
                total_weight = sum(asset_weights)
                if total_weight > 0:
                    asset_weights = [w / total_weight for w in asset_weights]
                
                portfolio_rets, raw_port_metrics, annual_port_metrics, risk_decomp_table = portfolio_metrics(
                    asset_data=asset_data,
                    weights=asset_weights,
                    asset_names=asset_names,
                    benchmark_data=(benchmark_data if not benchmark_data.empty else None),
                    risk_free_df=(risk_free_df if not risk_free_df.empty else None)
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
                    range=[
                        'blue', 'orange', 'green', 'purple', 'brown', 'red', 'pink', 'teal', 'gray', 'cyan',
                        'magenta', 'lime', 'indigo', 'yellow', 'coral', 'turquoise', 'gold', 'navy', 'olive', 'maroon'
                    ][:len(filtered_data['Label'].unique())]
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
                    
                    # --- New: Date Range Selector for Portfolio Growth ---
                    date_range = st.date_input(
                        "Select Date Range for Portfolio Growth Plot",
                        [portfolio_rets.index.min().date(), portfolio_rets.index.max().date()]
                    )
                    
                    # Validate the selected date range and filter portfolio returns
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        if start_date > end_date:
                            st.error("Start date must be before end date.")
                            st.stop()
                        filtered_portfolio_rets = portfolio_rets[
                            (portfolio_rets.index >= pd.to_datetime(start_date)) &
                            (portfolio_rets.index <= pd.to_datetime(end_date))
                        ]
                    else:
                        filtered_portfolio_rets = portfolio_rets

                    if len(asset_data) > 1:
                        portfolio_growth_df = pd.DataFrame({
                            'time': filtered_portfolio_rets.index,
                            'Growth': (1 + filtered_portfolio_rets).cumprod(),
                            'Label': 'Portfolio'
                        })
                        portfolio_growth_df['time'] = pd.to_datetime(portfolio_growth_df['time'], errors='coerce')
                        portfolio_growth_df = portfolio_growth_df.dropna(subset=['time']) \
                                                                .drop_duplicates(subset=['time']) \
                                                                .sort_values('time')
                        
                        if 'benchmark_data' in locals() and not benchmark_data.empty:
                            # Filter benchmark data using the same date range
                            benchmark_data['time'] = pd.to_datetime(benchmark_data['time'], errors='coerce')
                            benchmark_data_filtered = benchmark_data[
                                (benchmark_data['time'] >= pd.to_datetime(start_date)) &
                                (benchmark_data['time'] <= pd.to_datetime(end_date))
                            ]
                            benchmark_growth_df = pd.DataFrame({
                                'time': benchmark_data_filtered['time'],
                                'Growth': (1 + benchmark_data_filtered['return']).cumprod(),
                                'Label': 'Benchmark'
                            })
                            benchmark_growth_df = benchmark_growth_df.dropna(subset=['time']) \
                                                                    .drop_duplicates(subset=['time']) \
                                                                    .sort_values('time')
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
                                    
                    if asset_data and all(df is not None and not df.empty for df in asset_data):
                        st.write("Portfolio Drift & Rebalancing Simulation")
                        portfolio_drift_rebalancing_simulation(asset_data, asset_names, asset_weights)
                    else:
                        st.info("Upload data for all assets to view the portfolio drift simulation.")

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
                        # Compute the best and worst year based on portfolio returns
                        best_year = portfolio_returns_yearly.idxmax()
                        best_return = portfolio_returns_yearly.max()
                        worst_year = portfolio_returns_yearly.idxmin()
                        worst_return = portfolio_returns_yearly.min()

                        st.markdown(f"**Best Year:** {best_year} with a return of {best_return:.2%}")
                        st.markdown(f"**Worst Year:** {worst_year} with a return of {worst_return:.2%}")
                    else:
                        st.warning("Portfolio returns or benchmark data is not available.")
                except Exception as e:
                    st.error(f"Error generating the table: {e}")
                
                with st.expander("Dividend Decomposition Analysis", expanded=True):
                    st.header("Dividend Decomposition: Price vs. Dividend Contributions")
                    # Loop over each asset’s dataframe (asset_data) and corresponding name (asset_names)
                    for i, df in enumerate(asset_data):
                        if not df.empty:
                            st.markdown(f"### {asset_names[i]}")
                            plot_dividend_decomposition(df, asset_names[i])
                        else:
                            st.warning(f"No data available for {asset_names[i]} to perform dividend decomposition.")

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
                
                
                #########################################################################################################
                # After all your portfolio calculations, add:
                with st.expander("Additional Visualizations", expanded=True):

                    
                    st.subheader("Rolling Volatility of Portfolio")
                    st.write("This chart shows how the portfolio's annualized volatility has changed over time based on a 12-month rolling window. It helps you see periods of increased or decreased risk.")
                    # portfolio_rets is your computed portfolio returns series
                    if not portfolio_rets.empty:
                        plot_rolling_volatility(portfolio_rets, window=12)
                    
                    st.subheader("Annual Returns by Asset Over Time")
                    # asset_data and asset_names are assumed to be defined earlier.
                    plot_annual_returns_over_time(asset_data, asset_names)
                    
                    st.subheader("Rolling 1Y Annualized Returns for the Portfolio")
                    if not portfolio_rets.empty:
                        plot_rolling_annual_returns(portfolio_rets, window=12)
                    else:
                        st.warning("Portfolio returns are empty. Unable to compute rolling annualized returns.")
                
                    st.subheader("Return Distributions for Each Asset")
                    for i, df in enumerate(asset_data):
                        if not df.empty:
                            asset_return_series = df.set_index("time")["total_return"]
                            plot_return_distribution(asset_return_series, asset_label=asset_names[i])
                    
                    st.subheader("Correlation Heatmap of Assets")
                    st.write("""
                    This heatmap visualizes the correlation matrix of the assets' returns.
                    It helps identify which assets move together and can highlight potential diversification benefits.
                    """)
                    aligned_returns = align_asset_data(asset_data, asset_names)
                    if not aligned_returns.empty:
                        corr_df = aligned_returns.corr()
                        fig6, ax6 = plt.subplots(figsize=(8, 6), dpi=300)
                        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="Blues", ax=ax6)
                        ax6.set_title("Correlation Heatmap of Assets", fontsize=16)
                        plt.tight_layout()
                        st.pyplot(fig6)
                        buf6 = BytesIO()
                        fig6.savefig(buf6, format="png", dpi=300)
                        buf6.seek(0)
                        st.download_button(label="Download Correlation Heatmap (PNG)", data=buf6, file_name="correlation_heatmap.png", mime="image/png")
                    else:
                        st.warning("Aligned returns for correlation heatmap are empty.")
                    
                    st.subheader("Risk vs. Return Scatter Plot")
                    # Prepare a dictionary of annualized metrics for individual assets
                    annual_metrics_dict = {}
                    for i, df in enumerate(asset_data):
                        if not df.empty:
                            # Compute simple annualized metrics from monthly returns
                            vol_annual = df["total_return"].std() * np.sqrt(12)
                            mean_return_annual = (1 + df["total_return"].mean()) ** 12 - 1
                            annual_metrics_dict[asset_names[i]] = {
                                "Volatility (Annual)": vol_annual,
                                "Arithmetic Mean Return (Annual)": mean_return_annual
                            }
                    # portfolio metrics (annual_port_metrics) were computed earlier
                    plot_risk_return_scatter(annual_metrics_dict, annual_port_metrics)

                    st.subheader("Monte Carlo Simulation")
                    sim_horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=60, value=12, step=1)
                    n_sims = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

                    # Run the Monte Carlo simulation.
                    try:
                        sim_df, avg_sim, (worst_index, best_index) = monte_carlo_simulation_portfolio(aligned_returns, asset_weights, horizon=sim_horizon, n_sims=n_sims)
                    except Exception as e:
                        st.error(f"Monte Carlo simulation error: {e}")
                        sim_df = pd.DataFrame()

                    if not sim_df.empty:
                        # Prepare the simulation data in long format for Altair.
                        sim_df_reset = sim_df.reset_index(drop=True)
                        sim_df_reset["Simulation"] = sim_df_reset.index.astype(str)
                        sim_long = pd.melt(sim_df_reset, id_vars=["Simulation"], var_name="Month", value_name="Value")
                        sim_long["Month"] = sim_long["Month"].str.replace("Month_", "").astype(int)
                        
                        # Prepare data for best and worst simulation paths.
                        best_sim = sim_df_reset.loc[[best_index]].copy()
                        best_long = pd.melt(best_sim, id_vars=["Simulation"], var_name="Month", value_name="Value")
                        best_long["Month"] = best_long["Month"].str.replace("Month_", "").astype(int)
                        
                        worst_sim = sim_df_reset.loc[[worst_index]].copy()
                        worst_long = pd.melt(worst_sim, id_vars=["Simulation"], var_name="Month", value_name="Value")
                        worst_long["Month"] = worst_long["Month"].str.replace("Month_", "").astype(int)
                        
                        # Prepare data for the average simulation.
                        avg_df = pd.DataFrame({
                            "Month": np.arange(1, sim_df.shape[1] + 1),
                            "Value": avg_sim
                        })
                        
                        # Base layer: all simulation paths.
                        base = alt.Chart(sim_long).mark_line(color='lightgray', opacity=0.2, strokeWidth=0.5).encode(
                            x=alt.X("Month:Q", title="Month"),
                            y=alt.Y("Value:Q", title="Cumulative Portfolio Value"),
                            detail="Simulation:N"
                        )
                        
                        # Average simulation layer (thick black line).
                        avg_line = alt.Chart(avg_df).mark_line(color='black', strokeWidth=3).encode(
                            x=alt.X("Month:Q"),
                            y=alt.Y("Value:Q")
                        )
                        
                        # Best simulation layer (thick green line).
                        best_line = alt.Chart(best_long).mark_line(color='green', strokeWidth=3).encode(
                            x=alt.X("Month:Q"),
                            y=alt.Y("Value:Q")
                        )
                        
                        # Worst simulation layer (thick red line).
                        worst_line = alt.Chart(worst_long).mark_line(color='red', strokeWidth=3).encode(
                            x=alt.X("Month:Q"),
                            y=alt.Y("Value:Q")
                        )
                        
                        # Combine layers.
                        final_chart = alt.layer(base, avg_line, best_line, worst_line).properties(
                            title="Monte Carlo Simulation of Portfolio Returns"
                        )
                        
                        st.altair_chart(final_chart, use_container_width=True)
                        
                        # Compute and display summary statistics.
                        stats_df = compute_simulation_stats(sim_df)
                        st.markdown("### Simulation Summary Statistics")
                        st.dataframe(stats_df.style.format({"Value": "{:.2%}"}))
                        
                        # Optional: Provide a CSV download for the simulation stats.
                        csv_stats = stats_df.to_csv(index=False)
                        st.download_button(
                            label="Download Simulation Statistics as CSV",
                            data=csv_stats,
                            file_name="simulation_statistics.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Monte Carlo simulation produced no data.")


                with st.expander("Portfolio Optimization", expanded=True):
                    st.subheader("Optimized Portfolio Allocation (Maximizing Sharpe Ratio)")
                    st.write("""
                    This section calculates the optimal portfolio weights by maximizing the Sharpe ratio.
                    The optimized allocation aims to achieve the best risk-adjusted returns.
                    The comparison between current and optimized weights, along with the forecasted growth of the optimized portfolio,
                    helps in understanding potential improvements in portfolio performance.
                    """)
                    aligned_returns = align_asset_data(asset_data, asset_names)
                    if not aligned_returns.empty:
                        try:
                            risk_free_rate = 0.001  # Adjust this value as needed (e.g., monthly risk-free rate)
                            optimized_weights = maximize_sharpe_ratio(aligned_returns, risk_free_rate=risk_free_rate, allow_short=False)
                            optimized_df = pd.DataFrame({
                                "Asset": asset_names,
                                "Optimized Weight": optimized_weights
                            })
                            st.write("**Optimized Weights:**")
                            st.dataframe(optimized_df)
                            current_weights_df = pd.DataFrame({
                                "Asset": asset_names,
                                "Current Weight": asset_weights,
                                "Optimized Weight": optimized_weights
                            })
                            st.write("**Comparison: Current vs. Optimized Weights**")
                            st.dataframe(current_weights_df)
                            portfolio_returns_opt = aligned_returns.dot(optimized_weights)
                            st.write("Optimized Portfolio Cumulative Growth:")
                            st.line_chart((1 + portfolio_returns_opt).cumprod())
                        except Exception as e:
                            st.error(f"Optimization error: {e}")
                    else:
                        st.error("Asset return data is unavailable for optimization.")
                    
                    

                    


                
                with st.expander("Appendix: Exportable Factsheet Charts (Static)"):
                    sns.set_theme(style="whitegrid", palette="deep")

                    # 1. Growth of Each Asset vs. Benchmark
                    st.markdown("### Growth of Each Asset vs. Benchmark")
                    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
                    for i, df in enumerate(asset_data):
                        if not df.empty:
                            df_sorted = df.sort_values("time")
                            growth = (1 + df_sorted["total_return"]).cumprod()
                            ax1.plot(df_sorted["time"], growth, label=asset_names[i], linewidth=2)
                    if not benchmark_data.empty:
                        bench_sorted = benchmark_data.sort_values("time")
                        bench_growth = (1 + bench_sorted["return"]).cumprod()
                        ax1.plot(bench_sorted["time"], bench_growth, label="Benchmark", color="black", linestyle="--", linewidth=2)
                    ax1.set_title("Growth of Each Asset vs. Benchmark", fontsize=16)
                    ax1.set_xlabel("Time", fontsize=14)
                    ax1.set_ylabel("Cumulative Growth", fontsize=14)
                    ax1.legend(fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    buf1 = BytesIO()
                    fig1.savefig(buf1, format="png", dpi=300)
                    buf1.seek(0)
                    st.download_button(label="Download Growth Chart (PNG)", data=buf1, file_name="growth_chart.png", mime="image/png")

                    # 2. Portfolio Growth
                    st.markdown("### Portfolio Growth")
                    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
                    portfolio_growth = (1 + portfolio_rets).cumprod()
                    ax2.plot(portfolio_growth.index, portfolio_growth, color="navy", linewidth=2)
                    ax2.set_title("Portfolio Growth", fontsize=16)
                    ax2.set_xlabel("Time", fontsize=14)
                    ax2.set_ylabel("Portfolio Value", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    buf2 = BytesIO()
                    fig2.savefig(buf2, format="png", dpi=300)
                    buf2.seek(0)
                    st.download_button(label="Download Portfolio Growth Chart (PNG)", data=buf2, file_name="portfolio_growth.png", mime="image/png")

                    # 3. Portfolio Drift & Rebalancing Simulation
                    st.markdown("### Portfolio Drift & Rebalancing Simulation")
                    # Here we use a default rebalance period (e.g., 12 months). Ensure df_rebalanced is defined.
                    df_rebalanced, final_weights = simulate_rebalanced_portfolio(asset_data, asset_names, asset_weights, rebalance_period=12)
                    if not df_rebalanced.empty:
                        fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
                        ax3.plot(df_rebalanced["time"], df_rebalanced["portfolio_value"], color="purple", linewidth=2)
                        ax3.set_title("Portfolio Drift & Rebalancing Simulation", fontsize=16)
                        ax3.set_xlabel("Time", fontsize=14)
                        ax3.set_ylabel("Portfolio Value", fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        buf3 = BytesIO()
                        fig3.savefig(buf3, format="png", dpi=300)
                        buf3.seek(0)
                        st.download_button(label="Download Drift & Rebalancing Chart (PNG)", data=buf3, file_name="drift_rebalancing.png", mime="image/png")
                    else:
                        st.warning("Drift & Rebalancing simulation produced no data.")


                    

                    # 5. Rolling Volatility of Portfolio
                    st.markdown("### Rolling Volatility of Portfolio")
                    fig5, ax5 = plt.subplots(figsize=(10, 6), dpi=300)
                    rolling_vol = portfolio_rets.rolling(window=12).std() * np.sqrt(12)
                    ax5.plot(rolling_vol.index, rolling_vol, color="darkgreen", linewidth=2)
                    ax5.set_title("Rolling Volatility of Portfolio", fontsize=16)
                    ax5.set_xlabel("Time", fontsize=14)
                    ax5.set_ylabel("Annualized Volatility", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig5)
                    buf5 = BytesIO()
                    fig5.savefig(buf5, format="png", dpi=300)
                    buf5.seek(0)
                    st.download_button(label="Download Rolling Volatility Chart (PNG)", data=buf5, file_name="rolling_volatility.png", mime="image/png")


            except Exception as e:
                st.error(f"Portfolio calculation error: {str(e)}")
        else:
            st.info("Please upload data for all assets to view portfolio analytics.")
          
    elif page == "Page 2: Additional Analysis":
        page2_portfolio_structuring()
if __name__ == "__main__":
    main()
