# --- START OF FILE app.py ---
# Formula 1 Race Time Predictor
# Aplikacja przewidująca czasy wyścigowe kierowców F1 na podstawie czasów kwalifikacyjnych

import os
import fastf1  # Biblioteka do analizy danych F1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Ustawienia cache, sesji i parametrów modelu
CACHE_DIR = './fastf1_cache'  # Katalog cache dla danych F1

# Parametry sesji docelowej (wyścig, którego czasy chcemy przewidzieć)
TARGET_YEAR = 2023
TARGET_GP = 'Australia'
TARGET_SESSION_TYPE = 'R'  # R = Race (wyścig)

# Parametry sesji źródłowej (kwalifikacje, na podstawie których przewidujemy)
SOURCE_YEAR = 2024
SOURCE_GP = 'Australia'
SOURCE_SESSION_TYPE = 'Q'  # Q = Qualifying (kwalifikacje)

# Parametry modelu uczenia maszynowego
TEST_SPLIT_SIZE = 0.2  # 20% danych do walidacji
RANDOM_STATE_SPLIT = 42  # Ziarno losowości dla podziału danych
RANDOM_STATE_MODEL = 39  # Ziarno losowości dla modelu
N_ESTIMATORS = 100  # Liczba drzew w modelu Gradient Boosting
LEARNING_RATE = 0.1  # Współczynnik uczenia modelu

# Nazwy kolumn danych
FEATURE_COL_NAME = 'SourceSessionTime(s)'  # Kolumna z czasami kwalifikacyjnymi (dane wejściowe)
TARGET_COL_NAME = 'TargetRaceLapTime(s)'  # Kolumna z czasami wyścigowymi (dane wyjściowe)

# --- Helper Functions ---

def time_to_seconds(lap_time: pd.Timedelta | float | None) -> float | None:
    """
    Konwertuje obiekt czasu okrążenia (Timedelta) na sekundy.
    
    Args:
        lap_time: Czas okrążenia jako Timedelta, float lub None
        
    Returns:
        Czas w sekundach lub None dla wartości NaN
    """
    if pd.isna(lap_time):
        return np.nan
    if isinstance(lap_time, pd.Timedelta):
        return lap_time.total_seconds()
    if isinstance(lap_time, (int, float)):  # Na wypadek, gdyby już był liczbą
        return float(lap_time)
    return np.nan

def get_fastest_lap_per_driver(laps_data: pd.DataFrame | None, time_col_name: str) -> pd.DataFrame:
    """
    Znajduje najszybsze okrążenie dla każdego kierowcy używając groupby.
    
    Args:
        laps_data: DataFrame z danymi okrążeń
        time_col_name: Nazwa kolumny do użycia dla czasów okrążeń
        
    Returns:
        DataFrame z najszybszymi okrążeniami każdego kierowcy
    """
    if laps_data is None or laps_data.empty:
        print("  Ostrzeżenie [get_fastest]: Wejściowe dane okrążeń są puste.")
        # Zwróć pusty DataFrame z oczekiwanymi kolumnami i indeksem
        return pd.DataFrame(columns=[time_col_name], index=pd.Index([], name='Driver'))

    required_cols = ['Driver', 'LapTime']
    if not all(col in laps_data.columns for col in required_cols):
        print(f"  BŁĄD [get_fastest]: Brak wymaganych kolumn ({required_cols}).")
        print(f"  Dostępne kolumny: {laps_data.columns.tolist()}")
        return pd.DataFrame(columns=[time_col_name], index=pd.Index([], name='Driver'))

    try:
        # Pracuj na kopii, aby uniknąć SettingWithCopyWarning
        laps_copy = laps_data.copy()
        # Konwersja czasów okrążeń na sekundy
        laps_copy['LapTime(s)'] = laps_copy['LapTime'].apply(time_to_seconds)
        # Usunięcie wierszy z brakującymi danymi
        laps_cleaned = laps_copy.dropna(subset=['LapTime(s)', 'Driver'])

        if laps_cleaned.empty:
             print("  Ostrzeżenie [get_fastest]: Brak ważnych okrążeń po czyszczeniu NaN.")
             return pd.DataFrame(columns=[time_col_name], index=pd.Index([], name='Driver'))

        # Znajdź indeksy najszybszych okrążeń dla każdego kierowcy
        fastest_indices = laps_cleaned.groupby('Driver')['LapTime(s)'].idxmin()
        fastest_laps = laps_cleaned.loc[fastest_indices]

        if not fastest_laps.empty:
            fastest_laps = fastest_laps.rename(columns={'LapTime(s)': time_col_name})
            # Wybierz tylko potrzebne kolumny i ustaw indeks
            processed_df = fastest_laps.set_index('Driver')[[time_col_name]]
            print(f"  [get_fastest] Znaleziono najszybsze czasy dla {len(processed_df)} kierowców.")
            return processed_df
        else:
             print("  Ostrzeżenie [get_fastest]: Nie znaleziono najszybszych okrążeń po grupowaniu.")
             return pd.DataFrame(columns=[time_col_name], index=pd.Index([], name='Driver'))

    except KeyError as e:
         print(f"  Błąd KeyError [get_fastest]: {e}. Sprawdź nazwy kolumn.")
    except Exception as e:
        print(f"  Niespodziewany błąd [get_fastest]: {e}")

    return pd.DataFrame(columns=[time_col_name], index=pd.Index([], name='Driver'))

def load_f1_session(year: int, gp: str, session_type: str) -> fastf1.core.Session | None:
    """
    Pobiera i ładuje dane dla określonej sesji F1.
    
    Args:
        year: Rok sesji
        gp: Nazwa Grand Prix
        session_type: Typ sesji (Q = kwalifikacje, R = wyścig)
        
    Returns:
        Obiekt sesji FastF1 lub None w przypadku błędu
    """
    print(f"\nŁadowanie danych dla: Rok={year}, GP='{gp}', Sesja='{session_type}'")
    try:
        # Pobranie danych sesji z API Ergast
        session = fastf1.get_session(year, gp, session_type)
        # Ładowanie tylko potrzebnych danych (bez telemetrii, pogody, wiadomości)
        session.load(telemetry=False, weather=False, messages=False)
        print(f"-> Dane dla {session.event['EventName']} {year} - {session_type} załadowane.")
        return session
    except fastf1.ergast.ErgastConnectionError as e:
         print(f"  Błąd połączenia z Ergast API: {e}")
    except ValueError as e:
        print(f"  Błąd: Nie znaleziono sesji {year}/{gp}/{session_type} lub problem z danymi - {e}")
    except Exception as e:
        print(f"  Niespodziewany błąd ładowania ({year}, {gp}, {session_type}): {e}")
    return None

# --- Główna Funkcja Wykonawcza ---

def main():
    """
    Główny przepływ przetwarzania danych i budowy modelu.
    Proces obejmuje ładowanie danych, preprocessing, trening modelu, 
    predykcję i prezentację wyników.
    """

    # --- 1. Ładowanie Danych ---
    print("-" * 30)
    print("1. Ładowanie Danych")
    print("-" * 30)

    # Inicjalizacja i konfiguracja cache FastF1
    if not os.path.exists(CACHE_DIR):
        try: os.makedirs(CACHE_DIR); print(f"Katalog cache '{CACHE_DIR}' utworzony.")
        except OSError as e: print(f"Błąd tworzenia cache: {e}")
    try: fastf1.Cache.enable_cache(CACHE_DIR); print(f"Cache FastF1 włączony: {CACHE_DIR}")
    except Exception as e: print(f"Błąd włączania cache: {e}")

    # Ładowanie danych sesji docelowej (wyścig 2023) i źródłowej (kwalifikacje 2024)
    target_session = load_f1_session(TARGET_YEAR, TARGET_GP, TARGET_SESSION_TYPE)
    source_session = load_f1_session(SOURCE_YEAR, SOURCE_GP, SOURCE_SESSION_TYPE)

    if not target_session or not source_session:
        print("\nBŁĄD KRYTYCZNY: Nie udało się załadować jednej z wymaganych sesji. Zakończenie.")
        return

    # --- 2. Preprocessing ---
    print("\n" + "-" * 30)
    print("2. Preprocessing Danych")
    print("-" * 30)

    X_train, y_train, X_predict, predict_drivers = None, None, None, None

    # Pobranie danych o okrążeniach z sesji
    laps_target_raw = target_session.laps if hasattr(target_session, 'laps') else None
    laps_source_raw = source_session.laps if hasattr(source_session, 'laps') else None

    # Przygotuj X (najlepszy czas z sesji źródłowej - Quali 2024)
    # Te dane będą używane jako cechy wejściowe modelu
    fastest_source_times = get_fastest_lap_per_driver(laps_source_raw, FEATURE_COL_NAME)

    # Przygotuj Y (wszystkie czasy z sesji docelowej - Race 2023)
    # Te dane będą używane jako wartości docelowe do treningu modelu
    all_target_laps = pd.DataFrame()
    if laps_target_raw is not None and not laps_target_raw.empty and all(c in laps_target_raw.columns for c in ['Driver', 'LapTime']):
        all_target_laps = laps_target_raw[['Driver', 'LapTime']].copy()
        all_target_laps[TARGET_COL_NAME] = all_target_laps['LapTime'].apply(time_to_seconds)
        all_target_laps.dropna(subset=[TARGET_COL_NAME, 'Driver'], inplace=True)
        print(f"  [preprocess] Przetworzono {len(all_target_laps)} prawidłowych okrążeń docelowych (Y).")
    else:
        print("  Ostrzeżenie [preprocess]: Brak danych okrążeń docelowych (Y) lub wymaganych kolumn.")

    # Połącz X i Y dla treningu - łączymy dane kierowców z obu sesji
    if not fastest_source_times.empty and not all_target_laps.empty:
        fastest_source_times_reset = fastest_source_times.reset_index()
        training_data_merged = pd.merge(
            all_target_laps, fastest_source_times_reset, on='Driver', how='inner'
        )
        print(f"  [preprocess] Połączono dane treningowe: {len(training_data_merged)} wierszy dla {training_data_merged['Driver'].nunique()} kierowców.")

        if not training_data_merged.empty:
            # Przygotowanie danych treningowych (X) i docelowych (y)
            X_train = training_data_merged[[FEATURE_COL_NAME]]
            y_train = training_data_merged[TARGET_COL_NAME]
            print(f"  [preprocess] Przygotowano X_train (shape: {X_train.shape}) i y_train (shape: {y_train.shape})")
        else:
            print("  [preprocess] Brak danych treningowych po połączeniu.")
    else:
        print("  [preprocess] Nie można połączyć danych - brak przetworzonych danych X lub Y.")

    # Przygotuj X_predict (najlepsze czasy z sesji źródłowej)
    # Te dane będą używane do predykcji czasów wyścigowych
    if not fastest_source_times.empty:
         # Usuń NaN z DataFrame (choć funkcja get_fastest... powinna to robić)
        fastest_source_times.dropna(inplace=True)
        if not fastest_source_times.empty:
            X_predict = fastest_source_times[[FEATURE_COL_NAME]]
            predict_drivers = fastest_source_times.index.tolist()
            print(f"  [preprocess] Przygotowano X_predict (shape: {X_predict.shape}) dla {len(predict_drivers)} kierowców.")
        else:
            print("  [preprocess] Brak danych do predykcji po usunięciu NaN.")
    else:
        print("  [preprocess] Brak danych źródłowych do przygotowania X_predict.")

    # Sprawdzenie, czy mamy wszystkie potrzebne dane
    if X_train is None or y_train is None:
        print("\nBŁĄD KRYTYCZNY: Nie udało się przygotować danych treningowych. Zakończenie.")
        return
    if X_predict is None or predict_drivers is None:
        print("\nBŁĄD KRYTYCZNY: Nie udało się przygotować danych do predykcji. Zakończenie.")
        return
    
    # --- PODGLĄD DANYCH --- #
    print("\n=== Podgląd danych treningowych (training_data_merged) ===")
    if 'training_data_merged' in locals() and training_data_merged is not None:
        print(training_data_merged.head(10))   # pierwsze 10 wierszy do podglądu
        print(f"Liczba wierszy: {len(training_data_merged)} | Liczba kierowców: {training_data_merged['Driver'].nunique()}")
    else:
        print("Brak danych treningowych.")

    print("\n=== Podgląd X_train ===")
    if X_train is not None:
        print(X_train.head(10))

    print("\n=== Podgląd y_train ===")
    if y_train is not None:
        print(y_train.head(10))

    print("\n=== Podgląd danych do predykcji (X_predict) ===")
    if X_predict is not None:
        print(X_predict.head(10))

    

    # --- 3. Trening Modelu ---
    print("\n" + "-" * 30)
    print("3. Trening Modelu")
    print("-" * 30)

    model = None
    mae_val = None

    try:
        # Podział na zbiór treningowy i walidacyjny
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE_SPLIT
        )
        print(f"  Podzielono dane na trening ({len(X_train_split)}) i walidację ({len(X_val)}).")

        # Definicja i trening modelu Gradient Boosting Regressor
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, random_state=RANDOM_STATE_MODEL
        )
        print("  Trenowanie modelu Gradient Boosting Regressor...")
        model.fit(X_train_split, y_train_split)
        print("  Trening zakończony.")

        # Ocena modelu na zbiorze walidacyjnym
        y_pred_val = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        print(f"  Średni Błąd Bezwzględny (MAE) na walidacji: {mae_val:.4f} s")

    except ValueError as e:
         # W przypadku błędu podziału, trening na wszystkich danych
         print(f"  Ostrzeżenie: Błąd podczas podziału danych ({e}). Trenowanie na wszystkich danych.")
         try:
            model = GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, random_state=RANDOM_STATE_MODEL
            )
            model.fit(X_train, y_train)
            print("  Trening zakończony (na wszystkich danych).")
            mae_val = None # Brak oceny walidacyjnej
         except Exception as train_e:
             print(f"  Błąd podczas treningu na wszystkich danych: {train_e}")
             model = None
    except Exception as e:
        print(f"  Niespodziewany błąd podczas treningu: {e}")
        model = None

    if model is None:
        print("\nBŁĄD KRYTYCZNY: Nie udało się wytrenować modelu. Zakończenie.")
        return

    # --- 4. Predykcja ---
    print("\n" + "-" * 30)
    print("4. Predykcja Wyników")
    print("-" * 30)

    predicted_results_df = None

    try:
        # Użycie modelu do przewidywania czasów wyścigowych
        print(f"Predykcja dla {len(X_predict)} kierowców...")
        predicted_lap_times = model.predict(X_predict)

        if len(predicted_lap_times) == len(predict_drivers):
            # Tworzenie DataFrame z przewidywanymi wynikami
            results_data = {'Driver': predict_drivers, 'PredictedRaceTime(s)': predicted_lap_times}
            predicted_results_df = pd.DataFrame(results_data)
            # Sortowanie wyników od najszybszego do najwolniejszego
            predicted_results_df = predicted_results_df.sort_values(by='PredictedRaceTime(s)', ascending=True)
            predicted_results_df = predicted_results_df.reset_index(drop=True)
            predicted_results_df.index += 1  # Indeksowanie od 1 (pozycje)
            print("  Predykcja zakończona pomyślnie.")
        else:
            print(f"  Błąd: Niezgodność liczby wyników ({len(predicted_lap_times)}) i kierowców ({len(predict_drivers)}).")
    except Exception as e:
        print(f"  Niespodziewany błąd podczas predykcji: {e}")

    if predicted_results_df is None:
        print("\nBŁĄD KRYTYCZNY: Nie udało się przeprowadzić predykcji. Zakończenie.")
        return

    # --- 5. Prezentacja Wyników ---
    print("\n" + "-" * 30)
    print("5. Prezentacja Wyników")
    print("-" * 30)

    # Tabela z przewidywanymi wynikami
    print(f"\n--- Przewidywane Wyniki (na podstawie Q {SOURCE_YEAR} vs R {TARGET_YEAR}) ---")
    try:
        pd.set_option('display.max_rows', len(predicted_results_df) + 1)
        print(predicted_results_df.to_string())
        pd.reset_option('display.max_rows')
    except Exception as e: print(f"Błąd wyświetlania tabeli: {e}")

    # Wyświetlenie przewidywanego zwycięzcy
    if not predicted_results_df.empty:
        winner = predicted_results_df.iloc[0]['Driver']
        time = predicted_results_df.iloc[0]['PredictedRaceTime(s)']
        print(f"\nModel wskazuje jako 'najszybszego': {winner} ({time:.4f} s)")

    # Informacja o dokładności modelu
    if mae_val is not None:
        print(f"\nInformacja o modelu: MAE (walidacja) = {mae_val:.4f} s (~{mae_val:.2f} s błędu)")
    else:
        print("\nInformacja o modelu: MAE na zbiorze walidacyjnym nieobliczone.")

    # Generowanie wizualizacji wyników
    print("\nGenerowanie wizualizacji...")
    try:
        top_n = 10  # Liczba najlepszych kierowców do wizualizacji
        results_to_plot = predicted_results_df.head(top_n)
        plt.figure(figsize=(12, 7))
        # Wykres słupkowy z przewidywanymi czasami
        sns.barplot(x='PredictedRaceTime(s)', y='Driver', data=results_to_plot.iloc[::-1],
                    palette='viridis', hue='Driver', legend=False)
        plt.xlabel(f"Przewidywany 'Czas Okrążenia' (s) (Q{SOURCE_YEAR} vs R{TARGET_YEAR})")
        plt.ylabel("Kierowca")
        plt.title(f"Przewidywany Ranking TOP {top_n} - Model (Q{SOURCE_YEAR} vs R{TARGET_YEAR})")
        plt.tight_layout()
        plot_filename = f"predicted_ranking_EXP_{TARGET_YEAR}R_vs_{SOURCE_YEAR}Q.png"
        # Zapisanie wykresu do pliku
        plt.savefig(plot_filename)
        print(f"Wizualizacja zapisana: {plot_filename}")
        plt.show()  # Może nie działać w niektórych środowiskach
    except ImportError: print("  Ostrzeżenie: Brak matplotlib/seaborn. Zainstaluj: pip install matplotlib seaborn")
    except Exception as e: print(f"  Błąd wizualizacji: {e}")

# --- Uruchomienie Głównej Funkcji ---
if __name__ == "__main__":
    main()
    print("\n--- Koniec działania skryptu ---")

# --- END OF FILE app.py ---
