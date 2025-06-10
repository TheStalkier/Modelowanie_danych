# Aplikacja do Analizy Danych Giełdowych - Szczegółowa Dokumentacja

## 📈 Przegląd Projektu

Niniejsza aplikacja została stworzona w ramach projektu zaliczeniowego i stanowi kompleksowe narzędzie do analizy danych giełdowych. Aplikacja umożliwia wielopłaszczyznową analizę indeksów giełdowych z wykorzystaniem nowoczesnych technik wizualizacji danych i analizy technicznej.

## 🎯 Cel Edukacyjny

Projekt ma na celu:
- Demonstrację umiejętności tworzenia interaktywnych aplikacji webowych w Pythonie
- Implementację zaawansowanych technik analizy danych finansowych
- Wykorzystanie bibliotek do wizualizacji danych (Plotly, Matplotlib)
- Zastosowanie wzorców projektowych i dobrych praktyk programistycznych

## 🔧 Architektura Techniczna

### Wzorzec MVC
Aplikacja została zbudowana w oparciu o wzorzec Model-View-Controller:
- **Model**: Klasy do zarządzania danymi (load_data, calculate_technical_indicators)
- **View**: Komponenty interfejsu użytkownika (Streamlit widgets)
- **Controller**: Funkcje logiki biznesowej (create_price_chart, generate_comments)

### Optymalizacja Wydajności
- Wykorzystanie dekoratora `@st.cache_data` do cachowania operacji wczytywania danych
- Lazy loading - dane ładowane tylko przy pierwszym dostępie
- Optymalizacja zapytań pandas dla dużych zbiorów danych

## 📊 Wskaźniki Techniczne - Teoria i Implementacja

### Średnie Kroczące (Moving Averages)
Implementowane są trzy typy średnich kroczących:

```python
# Krótkoterminowa (20 dni) - sygnały szybkie, więcej szumu
data['MA20'] = data['Close'].rolling(window=20).mean()

# Średnioterminowa (50 dni) - balans między szybkością a stabilnością  
data['MA50'] = data['Close'].rolling(window=50).mean()

# Długoterminowa (200 dni) - trend główny, mniej szumu
data['MA200'] = data['Close'].rolling(window=200).mean()
```

**Interpretacja sygnałów:**
- Złoty krzyż: MA20 przecina MA50 od dołu → sygnał kupna
- Krzyż śmierci: MA20 przecina MA50 od góry → sygnał sprzedaży

### RSI (Relative Strength Index)
Wzór matematyczny RSI:
```
RSI = 100 - (100 / (1 + RS))
gdzie RS = Average Gain / Average Loss
```

Implementacja w kodzie:
```python
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
```

## 🎨 Komponenty Interfejsu Użytkownika

### Panel Boczny (Sidebar)
Implementuje zaawansowany system filtrowania:
```python
with st.sidebar:
    # Tryb wyboru indeksów
    selection_mode = st.radio(
        "Tryb wyboru indeksów:",
        ["Wybór indeksów", "Najlepsze/Najgorsze N"]
    )
    
    # Predefiniowane okresy czasowe
    date_option = st.radio(
        "Wybierz okres:",
        ["Niestandardowy", "Ostatni rok", "Ostatnie 6 miesięcy"]
    )
```

### System Zakładek
Aplikacja wykorzystuje pięć głównych zakładek:
1. **Trend cenowy** - wykresy cen i wolumenu
2. **Analiza techniczna** - wskaźniki techniczne z interpretacją
3. **Porównanie indeksów** - korelacje i porównania międzyrynkowe
4. **Statystyki** - miary statystyczne i eksport danych
5. **Komentarze** - automatycznie generowane analizy

## 🔍 Zaawansowane Funkcjonalności

### Automatyczne Generowanie Komentarzy
Aplikacja wykorzystuje algorytm analizy technicznej do generowania komentarzy:

```python
def generate_comments(data, ticker, start_date, end_date):
    """
    Generuje komentarze na podstawie:
    - Pozycji ceny względem średnich kroczących
    - Wartości RSI i jej interpretacji
    - Zmienności indeksu
    - Ogólnego trendu w analizowanym okresie
    """
```

### Analiza Korelacji
Macierz korelacji obliczana metodą Pearsona:
```python
pivot_data = filtered_data.pivot_table(
    index='Date', 
    columns='Ticker', 
    values='Pct_Change'
)
corr_matrix = pivot_data.corr()
```

### Porównanie Lat
Normalizacja danych do porównań względnych:
```python
first_close = data_year['Close'].iloc[0]
data_year['Normalized'] = data_year['Close'] / first_close * 100
```

## 🎯 Wartość Edukacyjna

### Umiejętności Techniczne
Projekt demonstruje znajomość:
- Programowania obiektowego w Pythonie
- Bibliotek do analizy danych (pandas, numpy)
- Narzędzi wizualizacji (Plotly, Matplotlib)
- Frameworka Streamlit do tworzenia aplikacji webowych
- Wzorców projektowych i architektury oprogramowania

### Wiedza Domenowa
Aplikacja pokazuje zrozumienie:
- Zasad analizy technicznej
- Interpretacji wskaźników finansowych
- Metodologii analizy danych czasowych
- Podstaw finansów i rynków kapitałowych

### Dokumentacja i Jakość Kodu
- Szczegółowe komentarze w kodzie
- Docstringi dla wszystkich funkcji
- Obsługa błędów i walidacja danych
- Responsywny design interfejsu

## 📋 Instrukcje Uruchomienia

### Krok 1: Przygotowanie Środowiska
```bash
# Utworzenie wirtualnego środowiska
python -m venv stock_analysis_env

# Aktywacja środowiska (Windows)
stock_analysis_env\Scripts\activate

# Aktywacja środowiska (Linux/Mac)
source stock_analysis_env/bin/activate
```

### Krok 2: Instalacja Zależności
```bash
pip install -r requirements.txt
```

### Krok 3: Uruchomienie Aplikacji
```bash
streamlit run stock_analysis_app.py
```

## 🔧 Możliwości Rozwoju

### Funkcjonalności do Dodania
1. **Prognozy cen** - wykorzystanie uczenia maszynowego
2. **Alerty** - powiadomienia o przekroczeniu poziomów technicznych
3. **Backtesting strategii** - testowanie strategii inwestycyjnych
4. **Więcej wskaźników** - MACD, Bollinger Bands, Stochastic
5. **Analiza fundamentalna** - integracja z danymi makroekonomicznymi

### Ulepszenia Techniczne
1. **Baza danych** - zastąpienie plików CSV bazą danych
2. **API finansowe** - real-time data feeds
3. **Deployment** - wdrożenie w chmurze (Heroku, AWS)
4. **Testy jednostkowe** - pokrycie kodu testami
5. **CI/CD** - automatyzacja procesów wdrożeniowych

## 📖 Bibliografia i Źródła

### Literatura Techniczna
- Murphy, John J. "Technical Analysis of the Financial Markets"
- Wilder, J. Welles Jr. "New Concepts in Technical Trading Systems"

### Dokumentacja Techniczna
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Datasety
- Kaggle: Stock Exchange Data
- Yahoo Finance API
- Alpha Vantage API

## 👨‍🎓 Wnioski Projektowe

Projekt demonstruje kompleksowe podejście do analizy danych finansowych, łącząc:
- **Aspekty techniczne** - programowanie, architektura, optymalizacja
- **Aspekty analityczne** - interpretacja danych, generowanie wniosków
- **Aspekty wizualne** - czytelne prezentowanie informacji
- **Aspekty użytkowe** - intuicyjny interfejs, responsywność

Aplikacja stanowi solidną podstawę do dalszego rozwoju w kierunku profesjonalnych narzędzi analizy finansowej i może służyć jako portfolio piece demonstrujące umiejętności programistyczne i analityczne.