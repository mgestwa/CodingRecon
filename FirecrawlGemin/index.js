// index.js - Główny plik serwera Express
const express = require('express');
const bodyParser = require('body-parser');
// Import FirecrawlApp zgodnie z aktualną dokumentacją
const { default: FirecrawlApp } = require('@mendable/firecrawl-js');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

// Konfiguracja middleware
app.use(bodyParser.json());
app.use(express.static('public'));

// Inicjalizacja klienta Firecrawl
const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_API_KEY
});

// Inicjalizacja klienta Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// Używanie modelu eksperymentalnego Gemini 2.5 Pro
const model = genAI.getGenerativeModel({ 
  model: "gemini-2.5-pro-exp-03-25", // Eksperymentalny model Gemini 2.5 Pro
  generationConfig: {
    temperature: 0.2,
    maxOutputTokens: 4000,
  }
});

// Endpoint do analizy strony internetowej
app.post('/api/analyze', async (req, res) => {
  try {
    const { url, query } = req.body;
    
    if (!url) {
      return res.status(400).json({ error: 'URL jest wymagany' });
    }
    
    console.log(`Rozpoczynanie analizy strony: ${url}`);
    
    // Scrapowanie strony za pomocą Firecrawl
    const scrapeResult = await firecrawl.scrapeUrl(url, {
      formats: ["markdown"]
    });
    
    console.log('Scrapowanie zakończone, przetwarzanie treści...');
    
    // Pobieranie treści markdown - sprawdzamy strukturę odpowiedzi
    let markdownContent = '';
    if (scrapeResult && scrapeResult.data && scrapeResult.data.markdown) {
      markdownContent = scrapeResult.data.markdown;
    } else if (scrapeResult && scrapeResult.markdown) {
      markdownContent = scrapeResult.markdown;
    } else {
      console.log('Nieoczekiwana struktura odpowiedzi:', JSON.stringify(scrapeResult).substring(0, 200) + '...');
      markdownContent = JSON.stringify(scrapeResult);
    }
    
    // Przygotowanie promptu dla Gemini z ograniczeniem długości tekstu
    const maxContentLength = 800000; // Limitowanie do ~800k znaków (by zmieścić się w kontekście Gemini)
    const truncatedContent = markdownContent.length > maxContentLength 
      ? markdownContent.substring(0, maxContentLength) + '... [treść obcięta]' 
      : markdownContent;
    
    // Określenie promptu dla Gemini w zależności od zapytania
    let prompt = 'Przeanalizuj poniższą treść strony i przygotuj krótkie podsumowanie najważniejszych informacji.';
    
    if (query) {
      prompt = `Przeanalizuj poniższą treść strony i odpowiedz na pytanie: "${query}". Podaj odpowiedź w oparciu o zawartość strony.`;
    }
    
    console.log(`Wysyłanie zapytania do modelu Gemini...`);
    
    try {
      // Format zgodny z aktualną dokumentacją API
      const geminiResponse = await model.generateContent({
        contents: [{ 
          role: 'user', 
          parts: [{ text: `${prompt}\n\nTreść strony:\n${truncatedContent}` }] 
        }]
      });
      
      const result = geminiResponse.response.text();
      
      // Zwrócenie wyników
      res.json({
        success: true,
        url,
        analysis: result,
        contentLength: markdownContent.length,
      });
    } catch (error) {
      console.error('Błąd podczas generowania odpowiedzi Gemini:', error);
      res.status(500).json({
        success: false,
        error: `Błąd modelu Gemini: ${error.message || 'Nieznany błąd'}`,
      });
    }
    
  } catch (error) {
    console.error('Błąd podczas analizy:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Wystąpił błąd podczas przetwarzania żądania',
    });
  }
});

// Uruchomienie serwera
app.listen(port, () => {
  console.log(`Serwer działa na porcie ${port}`);
});