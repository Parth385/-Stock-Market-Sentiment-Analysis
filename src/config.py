"""
Beyond Sentiment - Configuration and Master Definitions
Stage 2: Data Ingestion and Integration Pipeline
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_MARKET_DIR = RAW_DIR / "market"
RAW_ACTIONS_DIR = RAW_DIR / "actions"
RAW_NEWS_DIR = RAW_DIR / "news"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"

# Historical Date Boundary
START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

# 15 Approved Research Cohort Companies
COMPANIES = {
    "RELIANCE": {
        "name": "Reliance Industries Limited",
        "ticker": "RELIANCE",
        "isin": "INE002A01018",
        "sector": "Energy",
        "industry": "Oil & Gas / Retail / Telecom",
        "aliases": [
            "Reliance Industries", "Reliance", "RIL", "Mukesh Ambani", 
            "Jio", "Reliance Retail", "Reliance Jio"
        ]
    },
    "TCS": {
        "name": "Tata Consultancy Services Limited",
        "ticker": "TCS",
        "isin": "INE467B01029",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "aliases": [
            "Tata Consultancy Services", "TCS", "Tata Consultancy"
        ]
    },
    "INFY": {
        "name": "Infosys Limited",
        "ticker": "INFY",
        "isin": "INE009A01021",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "aliases": [
            "Infosys", "Infy", "Salil Parekh"
        ]
    },
    "HDFCBANK": {
        "name": "HDFC Bank Limited",
        "ticker": "HDFCBANK",
        "isin": "INE040A01034",
        "sector": "Financial Services",
        "industry": "Private Banking",
        "aliases": [
            "HDFC Bank", "HDFC Bk", "Sashidhar Jagdishan"
        ]
    },
    "ICICIBANK": {
        "name": "ICICI Bank Limited",
        "ticker": "ICICIBANK",
        "isin": "INE090A01021",
        "sector": "Financial Services",
        "industry": "Private Banking",
        "aliases": [
            "ICICI Bank", "ICICI Bk", "Sandeep Bakhshi"
        ]
    },
    "SBIN": {
        "name": "State Bank of India",
        "ticker": "SBIN",
        "isin": "INE062A01020",
        "sector": "Financial Services",
        "industry": "Public Sector Banking",
        "aliases": [
            "State Bank of India", "SBI", "State Bank"
        ]
    },
    "HINDUNILVR": {
        "name": "Hindustan Unilever Limited",
        "ticker": "HINDUNILVR",
        "isin": "INE030A01027",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Diversified FMCG",
        "aliases": [
            "Hindustan Unilever", "HUL", "Unilever India"
        ]
    },
    "ITC": {
        "name": "ITC Limited",
        "ticker": "ITC",
        "isin": "INE154A01025",
        "sector": "Fast Moving Consumer Goods",
        "industry": "FMCG / Cigarettes / Hotels / Paper",
        "aliases": [
            "ITC Limited", "ITC Ltd", "ITC", "Sanjiv Puri"
        ]
    },
    "LT": {
        "name": "Larsen & Toubro Limited",
        "ticker": "LT",
        "isin": "INE018A01030",
        "sector": "Construction & Infrastructure",
        "industry": "Engineering & Construction",
        "aliases": [
            "Larsen & Toubro", "L&T", "Larsen and Toubro"
        ]
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel Limited",
        "ticker": "BHARTIARTL",
        "isin": "INE397D01024",
        "sector": "Telecommunications",
        "industry": "Telecom Services",
        "aliases": [
            "Bharti Airtel", "Airtel", "Sunil Mittal"
        ]
    },
    "SUNPHARMA": {
        "name": "Sun Pharmaceutical Industries Limited",
        "ticker": "SUNPHARMA",
        "isin": "INE044A01036",
        "sector": "Healthcare & Pharmaceuticals",
        "industry": "Pharmaceuticals",
        "aliases": [
            "Sun Pharma", "Sun Pharmaceutical", "Dilip Shanghvi"
        ]
    },
    "CIPLA": {
        "name": "Cipla Limited",
        "ticker": "CIPLA",
        "isin": "INE059A01026",
        "sector": "Healthcare & Pharmaceuticals",
        "industry": "Pharmaceuticals",
        "aliases": [
            "Cipla Limited", "Cipla"
        ]
    },
    "MARUTI": {
        "name": "Maruti Suzuki India Limited",
        "ticker": "MARUTI",
        "isin": "INE585B01010",
        "sector": "Automobile",
        "industry": "Passenger Vehicles",
        "aliases": [
            "Maruti Suzuki", "Maruti", "RC Bhargava"
        ]
    },
    "M&M": {
        "name": "Mahindra & Mahindra Limited",
        "ticker": "M&M",
        "isin": "INE101A01026",
        "sector": "Automobile",
        "industry": "Automotive & Farm Equipment",
        "aliases": [
            "Mahindra & Mahindra", "Mahindra and Mahindra", "M&M", 
            "Anand Mahindra", "Mahindra"
        ]
    },
    "TATASTEEL": {
        "name": "Tata Steel Limited",
        "ticker": "TATASTEEL",
        "isin": "INE081A01020",
        "sector": "Metals & Mining",
        "industry": "Steel Manufacturing",
        "aliases": [
            "Tata Steel", "Tisco"
        ]
    }
}

TARGET_TICKERS = list(COMPANIES.keys())

# Sector mapping
SECTORS = {
    "Energy": ["RELIANCE"],
    "Information Technology": ["TCS", "INFY"],
    "Financial Services": ["HDFCBANK", "ICICIBANK", "SBIN"],
    "Fast Moving Consumer Goods": ["HINDUNILVR", "ITC"],
    "Construction & Infrastructure": ["LT"],
    "Telecommunications": ["BHARTIARTL"],
    "Healthcare & Pharmaceuticals": ["SUNPHARMA", "CIPLA"],
    "Automobile": ["MARUTI", "M&M"],
    "Metals & Mining": ["TATASTEEL"]
}

# Source URLs
TEJHQ_BASE_URL = "https://huggingface.co/datasets/tejhq/indian-markets/resolve/main"
MVEEN3_BASE_URL = "https://huggingface.co/datasets/mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker/resolve/main"
