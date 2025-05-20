# F1 Race Predictor

Aplikacja wykorzystująca uczenie maszynowe do przewidywania czasów wyścigowych kierowców Formuły 1 na podstawie danych kwalifikacyjnych.

## Opis projektu

F1 Race Predictor analizuje dane z sesji kwalifikacyjnych i wyścigów Formuły 1, aby przewidzieć, jak kierowcy poradziliby sobie w wyścigu na podstawie ich wyników kwalifikacyjnych. Aplikacja wykorzystuje model regresji Gradient Boosting do nauczenia się zależności między czasami kwalifikacyjnymi a wyścigowymi.

## Funkcjonalności

- Pobieranie danych sesji F1 (kwalifikacje i wyścigi) za pomocą biblioteki FastF1
- Przetwarzanie i czyszczenie danych czasów okrążeń
- Trenowanie modelu regresji Gradient Boosting
- Przewidywanie czasów wyścigowych na podstawie czasów kwalifikacyjnych
- Wizualizacja przewidywanych wyników w formie tabeli i wykresu

## Przykładowy scenariusz

Domyślne ustawienia aplikacji:
- Dane treningowe: Wyścig Australia 2023
- Dane do predykcji: Kwalifikacje Australia 2024
- Pytanie, na które odpowiada model: "Jak potoczyłby się wyścig w 2024 roku na podstawie czasów kwalifikacyjnych?"

## Wymagania

- Python 3.8+
- fastf1 (biblioteka do analizy danych F1)
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Instalacja

1. Sklonuj to repozytorium
2. Zainstaluj wymagane zależności:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
```

## Użycie

Uruchom aplikację:

```bash
python app.py
```

## Konfiguracja

Możesz dostosować parametry aplikacji edytując zmienne konfiguracyjne na początku pliku `app.py`:

- `TARGET_YEAR`, `TARGET_GP`, `TARGET_SESSION_TYPE` - rok, Grand Prix i typ sesji docelowej (wyścig)
- `SOURCE_YEAR`, `SOURCE_GP`, `SOURCE_SESSION_TYPE` - rok, Grand Prix i typ sesji źródłowej (kwalifikacje)
- Parametry modelu: `TEST_SPLIT_SIZE`, `N_ESTIMATORS`, `LEARNING_RATE` itp.

## Wyniki

Aplikacja generuje:
1. Tabelę z przewidywanymi czasami wyścigowymi dla każdego kierowcy
2. Informacje o przewidywanym zwycięzcy
3. Metryki jakości modelu (MAE)
4. Wizualizację w formie wykresu słupkowego z TOP 10 kierowców

## Struktura projektu

- `app.py` - główny plik aplikacji
- `fastf1_cache/` - katalog cache dla danych FastF1 (tworzony automatycznie)
- `predicted_ranking_*.png` - wygenerowane wykresy z przewidywanymi wynikami

## Zastrzeżenia

- Dokładność predykcji zależy od dostępności i jakości danych historycznych
- Model nie uwzględnia wielu zmiennych wpływających na rzeczywiste wyniki wyścigów (strategia pit stopów, pogoda, wypadki itp.)
- Aplikacja ma charakter demonstracyjny i edukacyjny

## Licencja

Ten projekt jest udostępniany na licencji MIT. 