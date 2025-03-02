\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny,
    keywordstyle=\color{blue},
}

\begin{document}

\title{Portfolio Analytics Dashboard Helper Documentation}
\author{AXSA Investment Partners}
\date{}
\maketitle

\section{Introduction}
This helper file provides detailed guidance on how to prepare and upload data for the Portfolio Analytics Dashboard. It specifies the required CSV file formats for your asset data and the optional risk-free rate data, along with the accepted date formats and the handling of dividends (whether as a yield or an actual amount). Following these instructions ensures your data is interpreted correctly by the dashboard when calculating total returns, drawdowns, and other metrics. This document also includes tips to enhance your experience and troubleshoot common issues, making it easier to leverage the dashboard’s full capabilities.

\section{Overview of Required Data}

\subsection{Asset Data}
Each asset you upload must be provided in a CSV file with the following columns (in exact names, without quotes):
\begin{itemize}
    \item \textbf{time}
    \item \textbf{price}
    \item \textbf{dividend}
    \item \textbf{return}
\end{itemize}

\noindent You must always include a \texttt{time} column. You may choose to provide both \texttt{price} and \texttt{return} or only one of them. The \texttt{dividend} column can be empty (all zeros or blanks) if no dividends are paid. Below is a description of each column:
\begin{itemize}
    \item \textbf{time:} The date/time associated with each entry. The dashboard expects monthly intervals but supports several date formats (see Section~\ref{sec:timeformats}).
    \item \textbf{price:} The asset’s price at the end of each month. If present, the dashboard will use it to calculate returns when combined with the \texttt{dividend} column.
    \item \textbf{dividend:} The dividend per share or a dividend yield percentage, depending on how you specify it. This column can be zero or left blank if dividends are not paid.
    \item \textbf{return:} The total return for that month. If you already include monthly total returns in this column (including dividends), you can leave \texttt{price} or \texttt{dividend} blank. If you provide all four columns, the system will use your \texttt{price} and \texttt{dividend} to compute total returns, ignoring the \texttt{return} column to avoid double counting.
\end{itemize}

\subsection{Risk-Free Rate Data }
You \textbf(must) provide a separate CSV for risk-free rates. It must contain:
\begin{itemize}
    \item \textbf{time} (or \textbf{date})
    \item \textbf{risk\_free\_rate}
\end{itemize}

\noindent For monthly data, each row should correspond to end-of-month dates, matching your asset data intervals. The \texttt{risk\_free\_rate} values must be numeric rates (e.g., \texttt{3.5} for 3.5\%) and are assumed to be annualized percentages, which the dashboard converts appropriately for monthly calculations.

\subsection{Benchmark Data (Optional)}
You can upload benchmark data via a CSV file or fetch it using a ticker/trading symbol which has data available on yahoo finance. If uploading a CSV, it should mirror the asset data format:
\begin{itemize}
    \item \textbf{time}
    \item \textbf{price}
    \item \textbf{dividend}
    \item \textbf{return}
\end{itemize}
If any column is missing, the dashboard will attempt to compute it (e.g., \texttt{return} from \texttt{price} or vice versa). This allows comparison of your portfolio against a benchmark like an index.

\section{Accepted Date Formats}
\label{sec:timeformats}
The date column (\texttt{time}) should be provided in one of the following formats:
\begin{itemize}
    \item \verb|YYYY-MM-DD| \quad (e.g., \texttt{2023-01-31})
    \item \verb|MM/DD/YYYY| \quad (e.g., \texttt{01/31/2023})
    \item \verb|DD-MM-YYYY| \quad (e.g., \texttt{31-01-2023})
    \item \verb|YYYY/MM/DD| \quad (e.g., \texttt{2023/01/31})
    \item \texttt{Month DD, YYYY} \quad (e.g., \texttt{January 31, 2023})
    \item ISO formats with time \quad (e.g., \texttt{2023-01-31 00:00:00})
\end{itemize}

\noindent The dashboard automatically converts all dates to the last day of the month (e.g., \texttt{2023-01-31}) for consistency, regardless of the day provided. For example, \texttt{2023-01-15} becomes \texttt{2023-01-31}. This ensures monthly alignment across all assets and benchmarks, avoiding mismatches in timing.

\section{Dividend Input Types}
The \texttt{dividend} column can represent either dividend yields (in percent) or actual cash amounts. You specify this in the dashboard’s sidebar under ``Dividend Input Type'':
\begin{enumerate}
    \item \textbf{Dividend as Yield:}
    \begin{itemize}
        \item Enter dividends as percentages (e.g., \texttt{2} for a 2\% yield).
        \item The system adds this yield to the price change to compute the total return.
        \item If \texttt{price} is absent, the system assumes \texttt{return} includes dividends when \texttt{dividend} is empty.
    \end{itemize}

    \item \textbf{Dividend as Actual Amount:}
    \begin{itemize}
        \item Enter cash amounts paid per share (e.g., \texttt{0.50} for \$0.50 per share).
        \item The system adds this amount to the change in \texttt{price} to compute the total return.
        \item If \texttt{price} is absent, \texttt{return} is used directly when \texttt{dividend} is empty.
    \end{itemize}
\end{enumerate}

\section{How the Dashboard Handles Data}

\subsection{Columns Provided}
\begin{itemize}
    \item \textbf{If you provide \texttt{time}, \texttt{return}, and \texttt{dividend}:}
    \begin{itemize}
        \item If \texttt{dividend} is zero or blank for all rows, the system uses your \texttt{return} as the total return (assuming it already factors in dividends).
        \item If \texttt{dividend} has non-zero values, the system assumes that \texttt{return} is only the price-change return. It reconstructs the price history internally and adds the \texttt{dividend} to generate total returns.
        \textcolor{red}{\textbf{Warning:}} This can double-count dividends if your \texttt{return} already includes them—check your data to avoid this.
    \end{itemize}

    \item \textbf{If you provide \texttt{time}, \texttt{price}, and \texttt{dividend}:}
    \begin{itemize}
        \item The total return is calculated directly from \texttt{price} changes plus \texttt{dividend}.
        \item Any \texttt{return} column present in the CSV is ignored to avoid confusion.
        \item If the \texttt{dividend} column is zero or blank, only price changes contribute to total return.
    \end{itemize}

    \item \textbf{If you provide all four columns (\texttt{time}, \texttt{price}, \texttt{dividend}, and \texttt{return}):}
    \begin{itemize}
        \item The system prioritizes \texttt{price} and \texttt{dividend} to compute total return.
        \item The existing \texttt{return} column is not used in calculations, preventing double dividend counting.
    \end{itemize}
\end{itemize}

\subsection{Risk-Free Rate Handling}
When you upload a \texttt{risk\_free\_rate} CSV with dates aligned to the end of each month:
\begin{itemize}
    \item The system matches your portfolio returns with the corresponding \texttt{risk\_free\_rate}.
    \item The risk-free rate is interpreted as an annualized percentage (e.g., \texttt{3.5} for 3.5\%, or 0.035 in decimal form), which is adjusted internally for metrics such as Sharpe Ratio and Alpha.
    \item Dates are aligned to asset data month-ends, with missing values forward-filled if necessary.
\end{itemize}

\subsection{Benchmark Handling}
\begin{itemize}
    \item \textbf{Ticker Option:} Enter a ticker (e.g., \textasciicircum GSPC) to fetch monthly data automatically from Yahoo Finance. The dashboard calculates returns and aligns them to your asset data.
    \item \textbf{CSV Option:} Upload a benchmark CSV with the same structure as asset data. The system ensures consistency by aligning dates and filling in missing columns as needed.
    \item \textbf{Global vs. Per-Asset:} Choose whether one benchmark applies to the whole portfolio or each asset has its own. Weights adjust the benchmark contribution accordingly.
\end{itemize}

\section{User Experience Tips}
\label{sec:ux_tips}
To make your interaction with the dashboard smooth and effective, consider these tips:

\subsection{Preparing Your CSV Files}
\begin{itemize}
    \item \textbf{Consistent Headers:} Use exact column names (\texttt{time}, \texttt{price}, \texttt{dividend}, \texttt{return}) to avoid mapping errors.
    \item \textbf{Monthly Consistency:} Ensure all rows represent month-end data. If dates vary (e.g., \texttt{2023-01-15} vs. \texttt{2023-01-31}), they’ll be standardized to the last day (e.g., \texttt{2023-01-31}), which might affect short periods.
    \item \textbf{Empty Dividends:} Leave \texttt{dividend} cells blank or set to \texttt{0} for non-dividend assets—no need to fill with dummy values.
\end{itemize}

\subsection{Common Pitfalls and Solutions}
\begin{itemize}
    \item \textbf{Double-Counting Dividends:} If your \texttt{return} includes dividends and \texttt{dividend} isn’t empty, you’ll see inflated returns. \\
    \emph{Solution:} Set \texttt{dividend} to zero if \texttt{return} is total return.
    \item \textbf{Missing Data:} Gaps in \texttt{price} or \texttt{return} can lead to NaN values in plots. \\
    \emph{Solution:} Fill gaps before uploading or ensure continuous monthly data.
    \item \textbf{Date Misalignment:} If asset and benchmark dates don’t overlap fully, some data might be excluded. \\
    \emph{Solution:} Check date ranges in your CSVs match your analysis period.
\end{itemize}

\subsection{Interacting with the Dashboard}
\begin{itemize}
    \item \textbf{Sidebar Navigation:} Use the sidebar to set the number of assets and upload files one-by-one. Expand each asset section to verify names and data.
    \item \textbf{Tabs:} Explore ``Portfolio Analysis'' for overall performance and ``Individual Asset Metrics'' for asset-specific insights. Use expanders to dive deeper into metrics like Monte Carlo simulations.
    \item \textbf{Feedback:} Watch for warnings (e.g., double-counting alerts) in the interface—they guide you to correct data issues on the fly.
\end{itemize}

\section{Troubleshooting Guide}
\label{sec:troubleshooting}
If you encounter issues, here’s how to resolve them:

\subsection{Error: ``Missing required columns''}
\begin{itemize}
    \item \textbf{Cause:} One or more of \texttt{time}, \texttt{price}, \texttt{dividend}, \texttt{return} isn’t in your CSV.
    \item \textbf{Fix:} Ensure all headers are present. If skipping \texttt{price} or \texttt{return}, include a blank column (e.g., \verb|,,0,|) to satisfy the requirement.
\end{itemize}

\subsection{Error: ``Portfolio calculation error''}
\begin{itemize}
    \item \textbf{Cause:} Data misalignment or NaNs in key columns.
    \item \textbf{Fix:} Verify all CSVs have consistent dates and no missing values in \texttt{time}. Fill NaNs in \texttt{price} or \texttt{return} with appropriate values.
\end{itemize}

\subsection{Unexpected Returns in Plots}
\begin{itemize}
    \item \textbf{Cause:} Dividend double-counting or mismatched \texttt{dividend} type.
    \item \textbf{Fix:} Check your \texttt{return} definition (price-only or total) and set \texttt{dividend} to zero if already included. Match ``Dividend Input Type'' to your CSV.
\end{itemize}

\section{Summary of Key Points}
\begin{itemize}
    \item \textbf{Columns:} Provide exactly these headers: \texttt{time}, \texttt{price}, \texttt{dividend}, \texttt{return}.
    \item \textbf{Monthly Data:} Rows should reflect monthly intervals. Dates are standardized to month-ends.
    \item \textbf{Date Formats:} Any format from Section~\ref{sec:timeformats} is accepted.
    \item \textbf{Dividends:} Specify yield (\%) or amount (\$); empty \texttt{dividend} is fine.
    \item \textbf{Risk-Free Rate/Benchmark:} Optional CSVs must align with asset dates.
    \item \textbf{Avoiding Double Counting:} Set \texttt{dividend} to zero if \texttt{return} includes dividends.
\end{itemize}

\section{Contact \& Support}
For further questions or troubleshooting, please reach out to the support team at:
\begin{itemize}
    \item Email: \href{mailto:support@example.com}{hriday@axsawm.com}
    
\end{itemize}

\end{document}
