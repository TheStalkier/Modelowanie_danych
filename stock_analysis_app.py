
# -*- coding: utf-8 -*-
"""
Aplikacja Streamlit do analizy danych giełdowych
================================================
Autorzy: Piech Faustyna; Skiba Maria;
Skrzek Martyna; Solarz Aleksandra;
Dawid Stachiewicz
Data zakończenia: 10.06.2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# Konfiguracja strony
st.set_page_config(
    page_title="Analiza Indeksów Giełdowych",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS do stylizacji
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# Funkcje pomocnicze
# ======================================================

@st.cache_data
def load_data():
    try:
        # Wczytanie danych z poprawką na nazwę kolumny
        df = pd.read_csv('indexData.csv').rename(columns={'Index': 'Ticker'})
        
        # Konwersja daty i sortowanie
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Obliczenia procentowych zmian
        df['Pct_Change'] = df.groupby('Ticker')['Close'].pct_change() * 100
        df['Cumulative_Return'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x / x.iloc[0] * 100 - 100
        )
        
        return df.dropna()
        
    except Exception as e:
        st.error(f"Błąd ładowania danych: {str(e)}")
        return None

@st.cache_data
def calculate_technical_indicators(ticker_data):
    """
    Oblicza wskaźniki techniczne dla danego indeksu.
    Zwraca nowy DataFrame z dodanymi wskaźnikami.
    """
    data = ticker_data.copy()
    
    # Średnie kroczące
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def create_price_chart(data, selected_tickers, show_ma=False):
    """
    Tworzy interaktywny wykres cen dla wybranych indeksów.

    Args:
        data (DataFrame): Dane indeksów
        selected_tickers (list): Lista wybranych indeksów
        show_ma (bool): Czy pokazać średnie kroczące

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Tworzenie wykresu
    fig = go.Figure()

    # Dodanie linii dla każdego wybranego indeksu
    for ticker in selected_tickers:
        ticker_data = data[data['Ticker'] == ticker]

        # Dodanie linii ceny zamknięcia
        fig.add_trace(go.Scatter(
            x=ticker_data['Date'],
            y=ticker_data['Close'],
            mode='lines',
            name=f'{ticker} - Cena zamknięcia',
            line=dict(width=2)
        ))

        # Dodanie średnich kroczących jeśli wybrano
        if show_ma:
            # MA20
            fig.add_trace(go.Scatter(
                x=ticker_data['Date'],
                y=ticker_data['MA20'],
                mode='lines',
                name=f'{ticker} - MA20',
                line=dict(width=1, dash='dot')
            ))

            # MA50
            fig.add_trace(go.Scatter(
                x=ticker_data['Date'],
                y=ticker_data['MA50'],
                mode='lines',
                name=f'{ticker} - MA50',
                line=dict(width=1, dash='dash')
            ))

    # Konfiguracja układu wykresu
    fig.update_layout(
        title='Analiza cen wybranych indeksów',
        xaxis_title='Data',
        yaxis_title='Cena',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    return fig

def create_performance_chart(data, selected_tickers, start_date, end_date):
    """
    Tworzy wykres porównawczy zmian procentowych dla wybranych indeksów.

    Args:
        data (DataFrame): Dane indeksów
        selected_tickers (list): Lista wybranych indeksów
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'].isin(selected_tickers)) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ].copy()

    # Obliczenie zmian procentowych względem pierwszego dnia dla każdego indeksu
    for ticker in selected_tickers:
        ticker_mask = filtered_data['Ticker'] == ticker
        first_close = filtered_data.loc[ticker_mask, 'Close'].iloc[0]
        filtered_data.loc[ticker_mask, 'Relative_Change'] = filtered_data.loc[ticker_mask, 'Close'] / first_close * 100 - 100

    # Tworzenie wykresu
    fig = px.line(
        filtered_data, 
        x='Date', 
        y='Relative_Change', 
        color='Ticker',
        labels={
            'Date': 'Data', 
            'Relative_Change': 'Zmiana procentowa (%)', 
            'Ticker': 'Indeks'
        },
        title='Porównanie zmian procentowych wybranych indeksów'
    )

    # Dodanie linii zerowej
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Konfiguracja układu wykresu
    fig.update_layout(
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    return fig

def create_rsi_chart(data, selected_tickers):
    """
    Tworzy wykres RSI dla wybranych indeksów.

    Args:
        data (DataFrame): Dane indeksów
        selected_tickers (list): Lista wybranych indeksów

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Tworzenie wykresu
    fig = go.Figure()

    # Dodanie linii RSI dla każdego wybranego indeksu
    for ticker in selected_tickers:
        ticker_data = data[data['Ticker'] == ticker]

        fig.add_trace(go.Scatter(
            x=ticker_data['Date'],
            y=ticker_data['RSI'],
            mode='lines',
            name=f'{ticker} - RSI'
        ))

    # Dodanie linii poziomów wykupienia/wyprzedania
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Wykupienie (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Wyprzedanie (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")

    # Konfiguracja układu wykresu
    fig.update_layout(
        title='Analiza RSI wybranych indeksów',
        xaxis_title='Data',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    return fig

def create_correlation_heatmap(data, selected_tickers, start_date, end_date):
    """
    Tworzy macierz korelacji dla wybranych indeksów.

    Args:
        data (DataFrame): Dane indeksów
        selected_tickers (list): Lista wybranych indeksów
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'].isin(selected_tickers)) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    # Tworzenie ramki danych z procentowymi zmianami cen dla każdego indeksu
    pivot_data = filtered_data.pivot_table(
        index='Date', 
        columns='Ticker', 
        values='Pct_Change'
    )

    # Obliczenie macierzy korelacji
    corr_matrix = pivot_data.corr()

    # Tworzenie wykresu cieplnego
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Macierz korelacji zmian procentowych indeksów'
    )

    # Konfiguracja układu wykresu
    fig.update_layout(
        xaxis_title='Indeks',
        yaxis_title='Indeks',
        template='plotly_white'
    )

    return fig

def create_candlestick_chart(data, ticker, start_date, end_date):
    """
    Tworzy wykres świecowy dla wybranego indeksu.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'] == ticker) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    # Tworzenie wykresu świecowego
    fig = go.Figure(data=[go.Candlestick(
        x=filtered_data['Date'],
        open=filtered_data['Open'],
        high=filtered_data['High'],
        low=filtered_data['Low'],
        close=filtered_data['Close'],
        name=ticker
    )])

    # Dodanie średnich kroczących
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['MA50'],
        mode='lines',
        name='MA50',
        line=dict(color='blue', width=1)
    ))

    # Konfiguracja układu wykresu
    fig.update_layout(
        title=f'Wykres świecowy dla {ticker}',
        xaxis_title='Data',
        yaxis_title='Cena',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )

    return fig

def create_technical_analysis_chart(data, ticker, start_date, end_date):
    """
    Tworzy wykres analizy technicznej z podwykresami dla wybranego indeksu.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'] == ticker) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    # Tworzenie podwykresów
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'Cena zamknięcia i średnie kroczące dla {ticker}', 'RSI (14)')
    )

    # Dodanie wykresu ceny i średnich kroczących
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Close'],
            mode='lines',
            name='Cena zamknięcia',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='green', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['MA200'],
            mode='lines',
            name='MA200',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )

    # Dodanie wykresu RSI
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )

    # Dodanie linii poziomów RSI
    fig.add_shape(
        type="line",
        x0=filtered_data['Date'].iloc[0],
        y0=70,
        x1=filtered_data['Date'].iloc[-1],
        y1=70,
        line=dict(color="red", width=1, dash="dash"),
        row=2, col=1
    )

    fig.add_shape(
        type="line",
        x0=filtered_data['Date'].iloc[0],
        y0=30,
        x1=filtered_data['Date'].iloc[-1],
        y1=30,
        line=dict(color="green", width=1, dash="dash"),
        row=2, col=1
    )

    # Konfiguracja układu wykresu
    fig.update_layout(
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    # Konfiguracja osi Y dla RSI
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig

def create_volume_analysis_chart(data, ticker, start_date, end_date):
    """
    Tworzy wykres analizy wolumenu dla wybranego indeksu.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'] == ticker) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    # Obliczenie średniego wolumenu
    avg_volume = filtered_data['Volume'].mean()

    # Tworzenie wykresu
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'Cena zamknięcia dla {ticker}', 'Wolumen')
    )

    # Dodanie wykresu ceny
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Close'],
            mode='lines',
            name='Cena zamknięcia',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Dodanie wykresu wolumenu
    # Kolor barek zależny od zmiany ceny
    colors = ['red' if row['Pct_Change'] < 0 else 'green' for _, row in filtered_data.iterrows()]

    fig.add_trace(
        go.Bar(
            x=filtered_data['Date'],
            y=filtered_data['Volume'],
            name='Wolumen',
            marker_color=colors
        ),
        row=2, col=1
    )

    # Dodanie linii średniego wolumenu
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=[avg_volume] * len(filtered_data),
            mode='lines',
            name='Średni wolumen',
            line=dict(color='black', width=1, dash='dash')
        ),
        row=2, col=1
    )

    # Konfiguracja układu wykresu
    fig.update_layout(
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    return fig

def create_yearly_comparison_chart(data, ticker, year1, year2):
    """
    Tworzy wykres porównawczy cen dla dwóch wybranych lat.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        year1 (int): Pierwszy rok do porównania
        year2 (int): Drugi rok do porównania

    Returns:
        plotly.graph_objects.Figure: Obiekt wykresu
    """
    # Filtracja danych dla dwóch lat
    data_year1 = data[
        (data['Ticker'] == ticker) & 
        (data['Date'].dt.year == year1)
    ].copy()

    data_year2 = data[
        (data['Ticker'] == ticker) & 
        (data['Date'].dt.year == year2)
    ].copy()

    # Dodanie dnia roku jako nowej kolumny
    data_year1['Day_of_Year'] = data_year1['Date'].dt.dayofyear
    data_year2['Day_of_Year'] = data_year2['Date'].dt.dayofyear

    # Normalizacja cen do pierwszego dnia
    if not data_year1.empty:
        first_close_1 = data_year1['Close'].iloc[0]
        data_year1['Normalized'] = data_year1['Close'] / first_close_1 * 100

    if not data_year2.empty:
        first_close_2 = data_year2['Close'].iloc[0]
        data_year2['Normalized'] = data_year2['Close'] / first_close_2 * 100

    # Tworzenie wykresu
    fig = go.Figure()

    # Dodanie linii dla pierwszego roku
    if not data_year1.empty:
        fig.add_trace(go.Scatter(
            x=data_year1['Day_of_Year'],
            y=data_year1['Normalized'],
            mode='lines',
            name=f'{ticker} - {year1}',
            line=dict(color='blue', width=2)
        ))

    # Dodanie linii dla drugiego roku
    if not data_year2.empty:
        fig.add_trace(go.Scatter(
            x=data_year2['Day_of_Year'],
            y=data_year2['Normalized'],
            mode='lines',
            name=f'{ticker} - {year2}',
            line=dict(color='red', width=2)
        ))

    # Konfiguracja układu wykresu
    fig.update_layout(
        title=f'Porównanie trendu {ticker} w latach {year1} i {year2}',
        xaxis_title='Dzień roku',
        yaxis_title='Znormalizowana cena (%)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )

    # Dodanie siatki dla kwartałów
    for q in [1, 91, 182, 273]:
        fig.add_vline(x=q, line_dash="dash", line_color="gray")

    return fig

def calculate_statistics(data, ticker, start_date, end_date):
    """
    Oblicza statystyki dla wybranego indeksu w określonym okresie.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        dict: Słownik ze statystykami
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'] == ticker) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    if filtered_data.empty:
        return None

    # Obliczenie statystyk
    stats = {
        'Początkowa cena': filtered_data['Close'].iloc[0],
        'Końcowa cena': filtered_data['Close'].iloc[-1],
        'Zmiana (%)': ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[0]) - 1) * 100,
        'Minimum': filtered_data['Close'].min(),
        'Maksimum': filtered_data['Close'].max(),
        'Średnia': filtered_data['Close'].mean(),
        'Mediana': filtered_data['Close'].median(),
        'Odchylenie standardowe': filtered_data['Close'].std(),
        'Zmienność (%)': (filtered_data['Close'].std() / filtered_data['Close'].mean()) * 100,
        'Średni dzienny zwrot (%)': filtered_data['Pct_Change'].mean(),
        'Najlepszy dzień (%)': filtered_data['Pct_Change'].max(),
        'Najgorszy dzień (%)': filtered_data['Pct_Change'].min(),
    }

    return stats

def generate_comments(data, ticker, start_date, end_date):
    """
    Generuje komentarze analityczne dla wybranego indeksu.

    Args:
        data (DataFrame): Dane indeksów
        ticker (str): Wybrany indeks
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa

    Returns:
        str: Komentarz analityczny
    """
    # Filtracja danych
    filtered_data = data[
        (data['Ticker'] == ticker) & 
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

    if filtered_data.empty:
        return "Brak danych do analizy."

    # Obliczenie podstawowych statystyk
    start_price = filtered_data['Close'].iloc[0]
    end_price = filtered_data['Close'].iloc[-1]
    total_change_pct = ((end_price / start_price) - 1) * 100

    # Średnie kroczące
    latest_close = filtered_data['Close'].iloc[-1]
    latest_ma20 = filtered_data['MA20'].iloc[-1]
    latest_ma50 = filtered_data['MA50'].iloc[-1]
    latest_ma200 = filtered_data['MA200'].iloc[-1]

    # RSI
    latest_rsi = filtered_data['RSI'].iloc[-1]

    # Przygotowanie komentarza
    comment = f"### Analiza {ticker} w okresie {start_date.date()} - {end_date.date()}"




    # Ogólny trend
    if total_change_pct > 0:
        comment += f"W analizowanym okresie indeks {ticker} wzrósł o **{total_change_pct:.2f}%**, "
        comment += f"od {start_price:.2f} do {end_price:.2f} punktów. "
        trend_description = "wzrostowy"
    else:
        comment += f"W analizowanym okresie indeks {ticker} spadł o **{abs(total_change_pct):.2f}%**, "
        comment += f"od {start_price:.2f} do {end_price:.2f} punktów. "
        trend_description = "spadkowy"

    # Średnie kroczące
    comment += "#### Analiza średnich kroczących"

    # Pozycja ceny względem średnich
    if latest_close > latest_ma20 > latest_ma50 > latest_ma200:
        comment += "Cena znajduje się **powyżej wszystkich średnich kroczących** (20, 50 i 200-sesyjnej), "
        comment += "co wskazuje na silny trend wzrostowy i potwierdza przewagę kupujących na rynku. "
    elif latest_close < latest_ma20 < latest_ma50 < latest_ma200:
        comment += "Cena znajduje się **poniżej wszystkich średnich kroczących** (20, 50 i 200-sesyjnej), "
        comment += "co wskazuje na silny trend spadkowy i potwierdza przewagę sprzedających na rynku. "
    elif latest_close > latest_ma50:
        comment += "Cena znajduje się **powyżej średniej 50-sesyjnej**, co może wskazywać na kontynuację trendu wzrostowego w średnim terminie. "
    elif latest_close < latest_ma50:
        comment += "Cena znajduje się **poniżej średniej 50-sesyjnej**, co może wskazywać na kontynuację trendu spadkowego w średnim terminie. "

    # Analiza przecięć średnich
    if latest_ma20 > latest_ma50 and filtered_data['MA20'].iloc[-20] < filtered_data['MA50'].iloc[-20]:
        comment += "W ostatnim czasie doszło do **przecięcia średniej 20-sesyjnej ze średnią 50-sesyjną** od dołu, "
        comment += "co jest interpretowane jako sygnał kupna (tzw. 'złoty krzyż' dla krótszych średnich). "
    elif latest_ma20 < latest_ma50 and filtered_data['MA20'].iloc[-20] > filtered_data['MA50'].iloc[-20]:
        comment += "W ostatnim czasie doszło do **przecięcia średniej 20-sesyjnej ze średnią 50-sesyjną** od góry, "
        comment += "co jest interpretowane jako sygnał sprzedaży (tzw. 'krzyż śmierci' dla krótszych średnich). "

    # RSI
    comment += "#### Analiza RSI"

    if latest_rsi > 70:
        comment += f"Wartość wskaźnika RSI wynosi obecnie **{latest_rsi:.2f}** i znajduje się w strefie wykupienia (powyżej 70), "
        comment += "co może sugerować, że indeks jest przewartościowany i możliwa jest korekta spadkowa. "
    elif latest_rsi < 30:
        comment += f"Wartość wskaźnika RSI wynosi obecnie **{latest_rsi:.2f}** i znajduje się w strefie wyprzedania (poniżej 30), "
        comment += "co może sugerować, że indeks jest niedowartościowany i możliwe jest odbicie wzrostowe. "
    else:
        comment += f"Wartość wskaźnika RSI wynosi obecnie **{latest_rsi:.2f}** i znajduje się w neutralnej strefie, "
        if latest_rsi > 50:
            comment += "z tendencją do wzrostu, co potwierdza ogólny trend wzrostowy. "
        else:
            comment += "z tendencją do spadku, co potwierdza ogólny trend spadkowy. "

    # Zmienność
    volatility = (filtered_data['Close'].std() / filtered_data['Close'].mean()) * 100
    comment += "#### Zmienność"
    comment += f"Zmienność indeksu w analizowanym okresie wynosi **{volatility:.2f}%**. "

    if volatility > 5:
        comment += "Jest to stosunkowo wysoka zmienność, co wskazuje na podwyższone ryzyko inwestycyjne, "
        comment += "ale jednocześnie oferuje potencjalne okazje inwestycyjne dla strategii krótkoterminowych. "
    else:
        comment += "Jest to stosunkowo niska zmienność, co wskazuje na stabilne zachowanie indeksu, "
        comment += "preferowane przy strategiach długoterminowych. "

    # Podsumowanie
    comment += "#### Podsumowanie"
    comment += f"Ogólny trend dla indeksu {ticker} w analizowanym okresie jest **{trend_description}**. "

    if trend_description == "wzrostowy" and latest_rsi < 70:
        comment += "Analiza techniczna wskazuje na kontynuację trendu wzrostowego, "
        comment += "a wskaźnik RSI nie sygnalizuje jeszcze wykupienia rynku. "
    elif trend_description == "wzrostowy" and latest_rsi > 70:
        comment += "Pomimo ogólnego trendu wzrostowego, wskaźnik RSI sygnalizuje wykupienie rynku, "
        comment += "co może prowadzić do krótkoterminowej korekty spadkowej. "
    elif trend_description == "spadkowy" and latest_rsi > 30:
        comment += "Analiza techniczna wskazuje na kontynuację trendu spadkowego, "
        comment += "a wskaźnik RSI nie sygnalizuje jeszcze wyprzedania rynku. "
    elif trend_description == "spadkowy" and latest_rsi < 30:
        comment += "Pomimo ogólnego trendu spadkowego, wskaźnik RSI sygnalizuje wyprzedanie rynku, "
        comment += "co może prowadzić do krótkoterminowego odbicia wzrostowego. "

    return comment

# ======================================================
# Główna aplikacja
# ======================================================

def main():
    """
    Główna funkcja aplikacji.
    """
    # Nagłówek aplikacji
    st.markdown('<h1 class="main-header">Analiza Indeksów Giełdowych</h1>', unsafe_allow_html=True)

    # Opis aplikacji
    with st.expander("📚 Opis aplikacji", expanded=False):
        st.markdown("""
        Ta aplikacja umożliwia analizę danych giełdowych z pliku indexData.csv. Możesz:
        - Przeglądać wykresy cen dla wybranych indeksów
        - Analizować wskaźniki techniczne (średnie kroczące, RSI)
        - Porównywać wydajność różnych indeksów
        - Generować statystyki i komentarze analityczne

        Dane są automatycznie przetwarzane przy wczytywaniu, a wszystkie wykresy są interaktywne.

        ### Korzystanie z aplikacji
        1. Wybierz indeksy do analizy w panelu bocznym
        2. Ustaw zakres dat
        3. Przejdź do zakładki z interesującym Cię rodzajem analizy
        4. Eksploruj interaktywne wykresy i statystyki

        ### Wskaźniki techniczne
        - **Średnie kroczące** - pomagają określić trend cenowy
        - **RSI (Relative Strength Index)** - wskazuje, czy rynek jest wykupiony (>70) lub wyprzedany (<30)
        """)

    # Wczytanie danych
    data = load_data()

    if data is None:
        st.error("Nie można kontynuować bez wczytania danych.")
        return

    # Panel boczny z filtrami
    with st.sidebar:
        st.markdown("## Filtry")

        # Wybór indeksów
        available_tickers = sorted(data['Ticker'].unique())

        # Tryb wyboru
        selection_mode = st.radio(
            "Tryb wyboru indeksów:",
            ["Wybór indeksów", "Najlepsze/Najgorsze N"]
        )

        if selection_mode == "Wybór indeksów":
            selected_tickers = st.multiselect(
                "Wybierz indeksy:",
                available_tickers,
                default=available_tickers[:3]  # Domyślnie pierwsze 3 indeksy
            )
        else:
            # Wybór najlepszych/najgorszych N
            top_n = st.number_input("Liczba indeksów:", min_value=1, max_value=len(available_tickers), value=3)

            sort_type = st.radio(
                "Sortowanie:",
                ["Najlepsze (rosnąco)", "Najgorsze (malejąco)"]
            )

            # Obliczenie zmian procentowych dla każdego indeksu
            ticker_changes = []
            for ticker in available_tickers:
                ticker_data = data[data['Ticker'] == ticker]
                start_price = ticker_data['Close'].iloc[0]
                end_price = ticker_data['Close'].iloc[-1]
                change_pct = ((end_price / start_price) - 1) * 100
                ticker_changes.append((ticker, change_pct))

            # Sortowanie według zmiany procentowej
            if sort_type == "Najlepsze (rosnąco)":
                ticker_changes.sort(key=lambda x: x[1], reverse=True)
            else:
                ticker_changes.sort(key=lambda x: x[1])

            # Wybór najlepszych/najgorszych N
            selected_tickers = [ticker for ticker, _ in ticker_changes[:top_n]]

            st.write("Wybrane indeksy:")
            for ticker, change in ticker_changes[:top_n]:
                st.write(f"- {ticker}: {change:.2f}%")

        # Wybór zakresu dat
        st.markdown("### Zakres dat")

        min_date = data['Date'].min().to_pydatetime()
        max_date = data['Date'].max().to_pydatetime()

        # Wybór predefiniowanych okresów
        date_option = st.radio(
            "Wybierz okres:",
            ["Niestandardowy", "Ostatni rok", "Ostatnie 6 miesięcy", "Ostatnie 3 miesiące", "Ostatni miesiąc"]
        )

        if date_option == "Niestandardowy":
            start_date = st.date_input(
                "Data początkowa:",
                min_value=min_date,
                max_value=max_date,
                value=min_date
            )

            end_date = st.date_input(
                "Data końcowa:",
                min_value=min_date,
                max_value=max_date,
                value=max_date
            )
        else:
            end_date = max_date

            if date_option == "Ostatni rok":
                start_date = end_date - timedelta(days=365)
            elif date_option == "Ostatnie 6 miesięcy":
                start_date = end_date - timedelta(days=182)
            elif date_option == "Ostatnie 3 miesiące":
                start_date = end_date - timedelta(days=91)
            else:  # Ostatni miesiąc
                start_date = end_date - timedelta(days=30)

            # Konwersja do formatu datetime.date dla wyświetlenia
            st.write(f"Data początkowa: {start_date.date()}")
            st.write(f"Data końcowa: {end_date.date()}")

        # Konwersja do datetime dla filtracji
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Wybór typu wizualizacji
        st.markdown("### Opcje wizualizacji")

        show_ma = st.checkbox("Pokaż średnie kroczące", value=True)

        st.markdown("---")
        st.markdown("### Informacje o aplikacji")
        st.markdown("""
        Aplikacja stworzona w ramach projektu zaliczeniowego.

        **Autorzy:** Piech Faustyna, Skiba Maria,
        Skrzek Martyna, Solarz Aleksandra,
        Dawid Stachiewicz  
        **Data zakończenia:** 10.06.2025
        """)

    # Sprawdzenie, czy wybrano indeksy
    if not selected_tickers:
        st.warning("Wybierz co najmniej jeden indeks w panelu bocznym.")
        return

    # Przygotowanie danych technicznych dla wybranych indeksów
    technical_data_list = []

    for ticker in selected_tickers:
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = calculate_technical_indicators(ticker_data)
        technical_data_list.append(ticker_data)

        # Aktualizacja głównego zbioru danych
        technical_data = pd.concat(technical_data_list, ignore_index=True)

    # Główny obszar z zakładkami
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend cenowy", 
        "🔍 Analiza techniczna", 
        "🔄 Porównanie indeksów",
        "📊 Statystyki",
        "📝 Komentarze"
    ])

    # Zakładka 1: Trend cenowy
    with tab1:
        st.markdown('<h2 class="sub-header">Analiza trendu cenowego</h2>', unsafe_allow_html=True)

        # Wybór typu wykresu
        chart_type = st.radio(
            "Typ wykresu:",
            ["Liniowy", "Świecowy"],
            horizontal=True
        )

        if chart_type == "Liniowy":
            # Wykres liniowy
            fig = create_price_chart(technical_data, selected_tickers, show_ma)
            st.plotly_chart(fig, use_container_width=True)

            # Wykres zmian procentowych
            st.markdown("### Porównanie zmian procentowych")
            fig_perf = create_performance_chart(technical_data, selected_tickers, start_date, end_date)
            st.plotly_chart(fig_perf, use_container_width=True, key="perf_chart_1")
        else:
            # Wykres świecowy (tylko dla jednego indeksu)
            if len(selected_tickers) > 1:
                ticker_for_candlestick = st.selectbox(
                    "Wybierz indeks dla wykresu świecowego:",
                    selected_tickers
                )
            else:
                ticker_for_candlestick = selected_tickers[0]

            fig = create_candlestick_chart(technical_data, ticker_for_candlestick, start_date, end_date)
            st.plotly_chart(fig, use_container_width=True)

        # Analiza wolumenu
        st.markdown("### Analiza wolumenu")

        if len(selected_tickers) > 1:
            ticker_for_volume = st.selectbox(
                "Wybierz indeks dla analizy wolumenu:",
                selected_tickers,
                key="volume_ticker"
            )
        else:
            ticker_for_volume = selected_tickers[0]

        fig_vol = create_volume_analysis_chart(technical_data, ticker_for_volume, start_date, end_date)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Zakładka 2: Analiza techniczna
    with tab2:
        st.markdown('<h2 class="sub-header">Analiza techniczna</h2>', unsafe_allow_html=True)

        # Wybór indeksu dla analizy technicznej
        if len(selected_tickers) > 1:
            ticker_for_ta = st.selectbox(
                "Wybierz indeks dla analizy technicznej:",
                selected_tickers
            )
        else:
            ticker_for_ta = selected_tickers[0]

        # Wykres z analizą techniczną
        fig_ta = create_technical_analysis_chart(technical_data, ticker_for_ta, start_date, end_date)
        st.plotly_chart(fig_ta, use_container_width=True)

        # Informacje o wskaźnikach
        with st.expander("ℹ️ Informacje o wskaźnikach", expanded=False):
            st.markdown("""
            ### Średnie kroczące

            **Średnia krocząca (MA)** jest wskaźnikiem trendu, który pokazuje średnią cenę z określonego okresu. Wygładza wahania cen, ułatwiając identyfikację trendu.

            - **MA20 (pomarańczowa)** - średnia z 20 sesji, odzwierciedla krótkoterminowy trend
            - **MA50 (zielona)** - średnia z 50 sesji, odzwierciedla średnioterminowy trend
            - **MA200 (czerwona)** - średnia z 200 sesji, odzwierciedla długoterminowy trend

            **Interpretacja:**
            - Cena powyżej średniej → trend wzrostowy
            - Cena poniżej średniej → trend spadkowy
            - Przecięcie średnich → potencjalna zmiana trendu
              - MA20 przecina MA50 od dołu → sygnał kupna (złoty krzyż)
              - MA20 przecina MA50 od góry → sygnał sprzedaży (krzyż śmierci)

            ### RSI (Relative Strength Index)

            **RSI** jest oscylatorem, który mierzy tempo i zmiany w ruchach cenowych. Waha się od 0 do 100 i pomaga identyfikować warunki wykupienia lub wyprzedania.

            **Interpretacja:**
            - RSI > 70 → rynek wykupiony, potencjalny sygnał sprzedaży
            - RSI < 30 → rynek wyprzedany, potencjalny sygnał kupna
            - RSI > 50 → ogólny trend wzrostowy
            - RSI < 50 → ogólny trend spadkowy
            """)

        # Analiza RSI
        st.markdown("### Analiza RSI")
        fig_rsi = create_rsi_chart(technical_data, [ticker_for_ta])
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Zakładka 3: Porównanie indeksów
    with tab3:
        st.markdown('<h2 class="sub-header">Porównanie indeksów</h2>', unsafe_allow_html=True)

        # Porównanie zmian procentowych
        st.markdown("### Porównanie zmian procentowych")
        fig_perf = create_performance_chart(technical_data, selected_tickers, start_date, end_date)
        st.plotly_chart(fig_perf, use_container_width=True, key="perf_chart_2")

        # Macierz korelacji
        st.markdown("### Macierz korelacji zmian procentowych")

        if len(selected_tickers) >= 2:
            fig_corr = create_correlation_heatmap(technical_data, selected_tickers, start_date, end_date)
            st.plotly_chart(fig_corr, use_container_width=True)

            with st.expander("ℹ️ Interpretacja macierzy korelacji", expanded=False):
                st.markdown("""
                **Macierz korelacji** pokazuje, jak silnie powiązane są zmiany cen różnych indeksów:

                - **+1.00** - idealna korelacja dodatnia (indeksy poruszają się identycznie)
                - **0.00** - brak korelacji (indeksy poruszają się niezależnie)
                - **-1.00** - idealna korelacja ujemna (indeksy poruszają się w przeciwnych kierunkach)

                **Interpretacja:**
                - **Wysoka korelacja dodatnia (>0.7)** - indeksy zazwyczaj poruszają się w tym samym kierunku
                - **Niska korelacja (-0.3 do 0.3)** - indeksy poruszają się w dużej mierze niezależnie
                - **Korelacja ujemna (<-0.3)** - indeksy często poruszają się w przeciwnych kierunkach

                Wiedza o korelacji jest przydatna przy:
                - Dywersyfikacji portfela (niższa korelacja = lepsza dywersyfikacja)
                - Analizie wpływu różnych rynków na siebie
                """)
        else:
            st.info("Wybierz co najmniej 2 indeksy, aby zobaczyć macierz korelacji.")

        # Porównanie lat
        st.markdown("### Porównanie trendów w różnych latach")

        available_years = sorted(technical_data['Date'].dt.year.unique())

        col1, col2 = st.columns(2)
        with col1:
            year1 = st.selectbox("Wybierz pierwszy rok:", available_years, index=len(available_years)-2)

        with col2:
            year2 = st.selectbox("Wybierz drugi rok:", available_years, index=len(available_years)-1)

        if len(selected_tickers) > 1:
            ticker_for_years = st.selectbox(
                "Wybierz indeks do porównania lat:",
                selected_tickers,
                key="years_comparison_ticker"
            )
        else:
            ticker_for_years = selected_tickers[0]

        fig_years = create_yearly_comparison_chart(technical_data, ticker_for_years, year1, year2)
        st.plotly_chart(fig_years, use_container_width=True)

        with st.expander("ℹ️ Jak interpretować wykres porównawczy lat", expanded=False):
            st.markdown("""
            Wykres porównawczy lat pokazuje, jak indeks zachowywał się w różnych latach, znormalizowany do 100 na początku każdego roku.

            **Co można zaobserwować:**
            - **Sezonowość** - powtarzające się wzory w tych samych okresach różnych lat
            - **Anomalie** - nietypowe zachowanie w jednym roku w porównaniu do drugiego
            - **Zmianę trendu** - różnice w ogólnym kierunku ruchu między latami

            **Praktyczne zastosowanie:**
            - Przewidywanie sezonowych wahań
            - Analiza wpływu ważnych wydarzeń gospodarczych
            - Identyfikacja długoterminowych cykli rynkowych
            """)

    # Zakładka 4: Statystyki
    with tab4:
        st.markdown('<h2 class="sub-header">Statystyki</h2>', unsafe_allow_html=True)

        # Wybór indeksu dla statystyk
        if len(selected_tickers) > 1:
            ticker_for_stats = st.selectbox(
                "Wybierz indeks dla statystyk:",
                selected_tickers,
                key="stats_ticker"
            )
        else:
            ticker_for_stats = selected_tickers[0]

        # Obliczenie statystyk
        stats = calculate_statistics(technical_data, ticker_for_stats, start_date, end_date)

        if stats:
            # Wyświetlenie podstawowych metryk
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Zmiana ceny",
                    value=f"{stats['Końcowa cena']:.2f}",
                    delta=f"{stats['Zmiana (%)']:.2f}%"
                )

            with col2:
                st.metric(
                    label="Min/Max",
                    value=f"{stats['Minimum']:.2f} / {stats['Maksimum']:.2f}"
                )

            with col3:
                st.metric(
                    label="Zmienność",
                    value=f"{stats['Zmienność (%)']:.2f}%"
                )

            # Tabela ze szczegółowymi statystykami
            st.markdown("### Szczegółowe statystyki")

            # Utworzenie tabeli statystyk
            stats_df = pd.DataFrame({
                'Miara': [
                    'Początkowa cena', 'Końcowa cena', 'Zmiana (%)',
                    'Minimum', 'Maksimum', 'Średnia', 'Mediana',
                    'Odchylenie standardowe', 'Zmienność (%)',
                    'Średni dzienny zwrot (%)', 'Najlepszy dzień (%)', 'Najgorszy dzień (%)'
                ],
                'Wartość': [
                    f"{stats['Początkowa cena']:.2f}",
                    f"{stats['Końcowa cena']:.2f}",
                    f"{stats['Zmiana (%)']:.2f}%",
                    f"{stats['Minimum']:.2f}",
                    f"{stats['Maksimum']:.2f}",
                    f"{stats['Średnia']:.2f}",
                    f"{stats['Mediana']:.2f}",
                    f"{stats['Odchylenie standardowe']:.2f}",
                    f"{stats['Zmienność (%)']:.2f}%",
                    f"{stats['Średni dzienny zwrot (%)']:.2f}%",
                    f"{stats['Najlepszy dzień (%)']:.2f}%",
                    f"{stats['Najgorszy dzień (%)']:.2f}%"
                ]
            })

            st.table(stats_df)

            # Eksport danych
            st.markdown("### Eksport danych")

            ticker_data_export = technical_data[technical_data['Ticker'] == ticker_for_stats].copy()
            ticker_data_export = ticker_data_export[
                (ticker_data_export['Date'] >= start_date) & 
                (ticker_data_export['Date'] <= end_date)
            ]

            st.download_button(
                label="Pobierz dane jako CSV",
                data=ticker_data_export.to_csv(index=False),
                file_name=f"{ticker_for_stats}_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Brak danych do obliczenia statystyk.")

    # Zakładka 5: Komentarze
    with tab5:
        st.markdown('<h2 class="sub-header">Komentarze analityczne</h2>', unsafe_allow_html=True)

        # Wybór indeksu dla komentarza
        if len(selected_tickers) > 1:
            ticker_for_comment = st.selectbox(
                "Wybierz indeks dla komentarza analitycznego:",
                selected_tickers,
                key="comment_ticker"
            )
        else:
            ticker_for_comment = selected_tickers[0]

        # Generowanie komentarza
        comment = generate_comments(technical_data, ticker_for_comment, start_date, end_date)

        # Wyświetlenie komentarza
        st.markdown(comment)

        # Dodatkowe informacje
        with st.expander("ℹ️ O komentarzach analitycznych", expanded=False):
            st.markdown("""
            **Komentarze analityczne** są generowane automatycznie na podstawie analizy danych technicznych. Obejmują:

            - **Ogólny trend** - kierunek ruchu ceny w analizowanym okresie
            - **Analizę średnich kroczących** - wzajemne położenie ceny i średnich
            - **Analizę RSI** - interpretacja aktualnej wartości wskaźnika
            - **Zmienność** - ocena zmienności indeksu i jej implikacje
            - **Podsumowanie** - synteza wszystkich wskaźników i ogólna rekomendacja

            Pamiętaj, że komentarze są generowane na podstawie analizy technicznej i nie uwzględniają czynników fundamentalnych ani zewnętrznych.
            """)

# Uruchomienie aplikacji
if __name__ == "__main__":
    main()
