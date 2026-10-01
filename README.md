# Crop Disease & Mandi Price Prediction System

This package is the project codebase. Your existing datasets are NOT included in this ZIP because they are large.

## Current dataset already prepared on your computer

Crop dataset:
G:\CROP PROJECT\FINAL\datasets\processed\crop

Disease dataset:
G:\CROP PROJECT\FINAL\datasets\processed\disease

The crop dataset currently contains 24 classes and 83,615 images.

## 1. Install

Open PowerShell:

    cd "G:\CROP PROJECT\FINAL"
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

If PowerShell blocks activation, run the Python commands using:
    .venv\Scripts\python.exe

## 2. Train crop model

    python training/train_crop_model.py

The model will be saved as:
    models/crop_classifier.pt

The first run downloads EfficientNet-B0 pretrained weights if they are not already cached.

## 3. Train disease models

    python training/train_disease_models.py

Models are saved under:
    models/disease/

Coconut is intentionally excluded because only 24 coconut disease images are currently available.

## 4. Mandi API

The official data.gov.in mandi resource used by this project is:
https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi

The resource ID is:
9ef84268-d588-465a-a308-a864a43d0070

Create a .env file by copying .env.example and put your API key in it.

Example:
DATA_GOV_API_KEY=YOUR_KEY

Do NOT commit the .env file.

## 5. Price prediction

Download historical mandi data for the commodity/market you want to model and save it as CSV.

The model expects at minimum:
arrival_date
modal_price

Prepare:

    python scripts/prepare_price_csv.py --input path\to\downloaded.csv

Train:

    python scripts/train_price_model.py --csv data/price/mandi_history.csv

This creates:
    models/price_model.joblib

The model uses lagged prices, rolling averages, recent change, month and day-of-week.

Important: future prices are estimates and should be displayed with appropriate uncertainty/limitations.

## 6. Run dashboard

    streamlit run app/app.py

Then open the local Streamlit URL shown in the terminal.

## Final pipeline

Image
 -> crop classifier
 -> crop-specific disease classifier
 -> latest available mandi price
 -> historical-price model
 -> dashboard

## Important limitation

This ZIP contains the complete project code/scaffold, not trained neural-network weights or the 83,615-image datasets. Those must remain on your computer and the models must be trained locally using the commands above.
