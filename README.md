# Aplikacja do Analizy Danych Giełdowych

## Opis projektu

Aplikacja Streamlit do analizy danych giełdowych z pliku indexData.csv, umożliwiająca interaktywną wizualizację i analizę indeksów giełdowych. Projekt został zrealizowany jako zaliczenie przedmiotu.

## Funkcjonalności

Aplikacja oferuje następujące funkcjonalności:

1. **Wczytywanie i przetwarzanie danych**:
   - Konwersja dat do formatu datetime
   - Filtrowanie indeksów
   - Obliczenie zmian procentowych cen
   - Obliczenie wskaźników technicznych

2. **Interfejs użytkownika**:
   - Panel boczny z filtrami:
     * Wybór indeksów (multiselect lub TOP N)
     * Zakres dat (predefiniowane okresy lub niestandardowy)
     * Opcje wizualizacji
   - Zakładki tematyczne z różnymi rodzajami analiz

3. **Rodzaje analiz**:
   - Trend cenowy (wykresy liniowe i świecowe)
   - Analiza techniczna (średnie kroczące i RSI)
   - Porównanie indeksów (korelacje, porównanie lat)
   - Statystyki (miary statystyczne, eksport danych)
   - Komentarze analityczne (automatycznie generowane)

4. **Wizualizacje**:
   - Interaktywne wykresy cen
   - Wykresy wskaźników technicznych
   - Macierze korelacji
   - Wykresy porównawcze lat

## Instalacja i uruchomienie

### Wymagania

Aplikacja wymaga Pythona 3.6+ oraz następujących bibliotek:
```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.15.0
```

### Instalacja

1. Sklonuj repozytorium lub pobierz pliki projektu
2. Zainstaluj wymagane biblioteki:
   ```
   pip install -r requirements.txt
   ```
3. Umieść plik `indexData.csv` w głównym katalogu projektu

### Uruchomienie

Uruchom aplikację poleceniem:
```
streamlit run stock_analysis_app.py
```

## Struktura projektu

```
├── stock_analysis_app.py   # Główny plik aplikacji
├── indexData.csv           # Plik z danymi indeksów giełdowych
├── README.md               # Dokumentacja projektu
└── requirements.txt        # Lista wymaganych bibliotek
```

## Szczegóły implementacji

### Przetwarzanie danych

Dane są wczytywane z pliku CSV i przetwarzane za pomocą funkcji `load_data()`:
```python
@st.cache_data
def load_data():
    """
    Wczytuje dane z pliku CSV i przeprowadza wstępne przetwarzanie.

    Returns:
        DataFrame: Wczytane i przetworzone dane
    """
    try:
        # Wczytanie danych z pliku CSV
        df = pd.read_csv('indexData.csv')

        # Konwersja kolumny daty na format datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Obliczenie zmian procentowych
        df['Pct_Change'] = df.groupby('Ticker')['Close'].pct_change() * 100

        # Inne przetwarzanie...

        return df

    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania danych: {e}")
        return None
```

### Wskaźniki techniczne

Aplikacja oblicza następujące wskaźniki techniczne:

1. **Średnie kroczące**:
   - MA20 (krótkoterminowa)
   - MA50 (średnioterminowa)
   - MA200 (długoterminowa)

2. **RSI (Relative Strength Index)**:
   - Obliczany na podstawie 14-dniowych zmian cen
   - Interpretacja: >70 (wykupienie), <30 (wyprzedanie)

```python
def calculate_technical_indicators(ticker_data):
    """
    Oblicza wskaźniki techniczne dla danego indeksu.
    """
    # Kopie danych
    data = ticker_data.copy()

    # Obliczenie średnich kroczących
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Obliczenie RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data
```

### Wizualizacje

Aplikacja wykorzystuje bibliotekę Plotly do tworzenia interaktywnych wykresów:

```python
def create_price_chart(data, selected_tickers, show_ma=False):
    """
    Tworzy interaktywny wykres cen dla wybranych indeksów.
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
            # MA20, MA50...

    # Konfiguracja układu wykresu...

    return fig
```

### Komentarze analityczne

Aplikacja generuje automatyczne komentarze analityczne na podstawie wskaźników technicznych:

```python
def generate_comments(data, ticker, start_date, end_date):
    """
    Generuje komentarze analityczne dla wybranego indeksu.
    """
    # Filtracja danych...

    # Obliczenie podstawowych statystyk...

    # Analiza średnich kroczących...

    # Analiza RSI...

    # Podsumowanie...

    return comment
```

## Dodatkowe funkcjonalności i rozszerzenia

1. **Wykresy porównawcze lat**:
   - Porównanie zachowania indeksu w różnych latach
   - Normalizacja do początku roku dla lepszego porównania

2. **Macierz korelacji**:
   - Analiza korelacji między zmianami cen różnych indeksów
   - Pomocna przy dywersyfikacji portfela

3. **Eksport danych**:
   - Możliwość pobrania przefiltrowanych danych w formacie CSV

4. **Responsywny design**:
   - Aplikacja dostosowuje się do różnych rozmiarów ekranu
   - Wykresy wykorzystują pełną szerokość kontenera

## Autor

Student  
Data: 10.06.2025

## Licencja

Ten projekt jest udostępniany na licencji MIT.
