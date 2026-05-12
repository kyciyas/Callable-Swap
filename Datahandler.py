import requests
import yfinance as yf
import QuantLib as ql
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

class Datahandler:
    def __init__(self, country="KR"):
        self.country = country
        if country == "KR":
            self.calendar = ql.SouthKorea()
            self.day_count = ql.Actual365Fixed()
            self.settlement_days = 1
            self.business_convention = ql.ModifiedFollowing
        elif country == "US":
            self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            self.day_count = ql.Actual360()
            self.settlement_days = 0
            self.business_convention = ql.Following
        else:
            raise Exception('KR is for Korean market and US is for US market.')

    def fetch_from_yfinance(self):
        tickers = {"1Y": "^IRX", "5Y": "^FVX", "10Y": "^TNX"}
        try:
            df = yf.download(list(tickers.values()), period="1y", progress=False)

            if df.empty or 'Close' not in df:
                raise ValueError("No data found")

            raw_series = df['Close'].ffill() / 100

            inv_tickers = {v: k for k, v in tickers.items()}  # {'^IRX': '1Y', ...}
            series_data = raw_series.rename(columns=inv_tickers)

            snapshot = {k: float(series_data[k].iloc[-1]) for k in tickers.keys()}

            return snapshot, series_data

        except Exception as e:
            print(f"yfinance error: {e}. Using fallback rates.")
            fallback_rates = {"1Y": 0.048, "5Y": 0.042, "10Y": 0.043}
            fallback_df = pd.DataFrame([fallback_rates], index=[datetime.now()])
            return fallback_rates, fallback_df

    def fetch_from_ecos(self, api_key=""):
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

        tickers = {"1Y": "010190000", "5Y": "010210000", "10Y": "010240000"}

        rates = {}
        rates_dict = {}
        for label, code in tickers.items():
            url = f"""
            https://ecos.bok.or.kr/api/StatisticSearch/
            {api_key}/json/kr/1/5000/
            817Y002/D/{start_date}/{end_date}/{code}
            """
            url = url.replace("\n", "").replace(" ", "")

            response = requests.get(url)
            res_data = response.json()

            rows = res_data['StatisticSearch']['row']
            last_val = rows[-1]['DATA_VALUE']
            rates[label] = float(last_val) / 100
            rates_dict[label] = [float(r['DATA_VALUE']) / 100 for r in rows]
            # print(f"Successfully fetched {label}: {last_val}%")

        return rates, rates_dict

    def get_calendar(self):
        return self.calendar

    def get_settlement_days(self):
        return self.settlement_days