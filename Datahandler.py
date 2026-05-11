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
            df = yf.download(list(tickers.values()), period="5d", progress=False)

            if df.empty or 'Close' not in df:
                raise ValueError("No data found")

            data = df['Close'].ffill().iloc[-1]
            return {k: float(data[v]) / 100 for k, v in tickers.items()}

        except Exception as e:
            print(f"yfinance error: {e}. Using fallback rates.")
            # 데이터 못 가져올 때를 대비한 백업 금리 (현재 시장 상황 비슷하게)
            return {"1Y": 0.048, "5Y": 0.042, "10Y": 0.043}

    def fetch_from_ecos(self, api_key=""):
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

        tickers = {"1Y": "010190000", "5Y": "010210000", "10Y": "010240000"}

        rates = {}
        for label, code in tickers.items():
            url = f"""
            https://ecos.bok.or.kr/api/StatisticSearch/
            {api_key}/json/kr/1/5000/
            817Y002/D/{start_date}/{end_date}/{code}
            """
            url = url.replace("\n", "").replace(" ", "")
            try:
                response = requests.get(url)
                res_data = response.json()

                if 'StatisticSearch' in res_data:
                    rows = res_data['StatisticSearch']['row']
                    last_val = rows[-1]['DATA_VALUE']
                    rates[label] = float(last_val) / 100
                    # print(f"Successfully fetched {label}: {last_val}%")
                else:
                    print(f"ECOS API Response Error ({label}): {res_data}")
                    rates[label] = 0.035
            except Exception as e:
                print(f"Error fetching {label}. Check your network or URL.")
                print(f"Detail: {e}")
                rates[label] = 0.035

        return rates

    def get_calendar(self):
        return self.calendar

    def get_settlement_days(self):
        return self.settlement_days