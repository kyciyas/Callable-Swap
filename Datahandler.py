import requests
import yfinance as yf
import QuantLib as ql
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas_datareader.data as web


class Datahandler:
    def __init__(self, country="KR"):
        self.country = country
        if country == "KR":
            self.calendar = ql.SouthKorea()
            self.day_count = ql.Actual365Fixed()
            self.settlement_days = 1
            self.business_convention = ql.ModifiedFollowing
            self.ois_convention = ql.Following
        elif country == "US":
            self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            self.day_count = ql.Actual360()
            self.settlement_days = 0
            self.business_convention = ql.Following
            self.ois_convention = ql.Following
        else:
            raise Exception('KR is for Korean market and US is for US market.')

    def fetch_from_yfinance(self):
        tickers = {"1Y": "^IRX", "5Y": "^FVX", "10Y": "^TNX"}
        raw_fixed_rate_from_yfinance = yf.download(list(tickers.values()), period="1y", progress=False)
        if raw_fixed_rate_from_yfinance.empty or 'Close' not in raw_fixed_rate_from_yfinance:
            raise ValueError("No data found")

        raw_series = raw_fixed_rate_from_yfinance['Close'].ffill() / 100
        inv_tickers = {v: k for k, v in tickers.items()}

        #Tickers conversion to match ECOS data
        series_data = raw_series.rename(columns=inv_tickers)
        snapshot = {k: float(series_data[k].iloc[-1]) for k in tickers.keys()}

        return snapshot, series_data

    def fetch_from_ecos(self, api_key="sample"):
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

        tickers = {"1Y": "010190000", "5Y": "010210000", "10Y": "010240000"}

        rates = {}
        rates_dict = {}
        for label, code in tickers.items():
            url = f"""https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/5000/817Y002/D/{start_date}/{end_date}/{code}"""
            url = url.replace("\n", "").replace(" ", "")

            response = requests.get(url)
            res_data = response.json()

            rows = res_data['StatisticSearch']['row']
            last_val = rows[-1]['DATA_VALUE']
            rates[label] = float(last_val) / 100
            rates_dict[label] = [float(r['DATA_VALUE']) / 100 for r in rows]

        return rates, rates_dict

    def get_calendar(self):
        return self.calendar

    def get_settlement_days(self):
        return self.settlement_days

    def get_days_count(self):
        return self.day_count

class OisDataHandler:
    def __init__(self, tag='KR', rates_dict = {}):
        self.country_tag = tag
        self.rates_dict = rates_dict
        if self.country_tag == 'KR':
            self.calendar = ql.SouthKorea(ql.SouthKorea.KRX)
            self.day_counter = ql.Actual365Fixed()
            self.settlement_days = 1
            self.day_counter = ql.Actual365Fixed()

        elif self.country_tag == 'US':
            self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            self.day_counter = ql.Actual360()
            self.settlement_days = 0
            self.day_counter = ql.Actual360()

    def fetch_live_kofr(self, api_key="sample") -> float:
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=8)).strftime('%Y%m%d')

        url = f"""https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/5000/817Y002/D/{start_date}/{end_date}/010503000"""
        url = url.replace("\n", "").replace(" ", "")

        response = requests.get(url, timeout=8)
        if response.status_code != 200:
            raise ConnectionError(f"HTTP 에러 상태코드: {response.status_code}")

        json_data = response.json()
        if "StatisticSearch" in json_data and "row" in json_data["StatisticSearch"]:
            latest_row = json_data["StatisticSearch"]["row"][-1]
            kofr_percent = float(latest_row["DATA_VALUE"])

            # print(f"[데이터] 한국은행 KOFR 무위험 기저금리 수집 완료: {kofr_percent}%")
            return kofr_percent / 100.0

        raise ValueError("JSON 응답에 StatisticSearch/row 노드가 누락되었습니다.")

    def fetch_live_sofr(self) -> float:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=150)

        df_sofr = web.DataReader('SOFR', 'fred', start_date, end_date)
        df_sofr = df_sofr.dropna()

        latest_date = df_sofr.index[-1].strftime('%Y-%m-%d')
        latest_rate = df_sofr['SOFR'].iloc[-1]
        # print(f"\n[{latest_date}] 기준 실제 SOFR 무위험 금리: {latest_rate}%")
        return latest_rate / 100

    def build_ois_curve(self, evaluation_date: ql.Date, rate = 0.035) -> ql.RelinkableYieldTermStructureHandle:
        ql.Settings.instance().evaluationDate = evaluation_date

        if self.country_tag == 'KR':
            overnight_index = ql.Kofr()
        else:
            overnight_index = ql.Sofr()

        latest_1y_rate = float(self.rates_dict['1Y'][-1])
        latest_5y_rate = float(self.rates_dict['5Y'][-1])
        latest_10y_rate = float(self.rates_dict['10Y'][-1])

        ois_helpers = [
            ql.OISRateHelper(self.settlement_days, ql.Period(1, ql.Years),
                             ql.QuoteHandle(ql.SimpleQuote(latest_1y_rate)), overnight_index),
            ql.OISRateHelper(self.settlement_days, ql.Period(5, ql.Years),
                             ql.QuoteHandle(ql.SimpleQuote(latest_5y_rate)), overnight_index),
            ql.OISRateHelper(self.settlement_days, ql.Period(10, ql.Years),
                             ql.QuoteHandle(ql.SimpleQuote(latest_10y_rate)), overnight_index)
        ]

        base_curve = ql.PiecewiseLogLinearDiscount(evaluation_date, ois_helpers, self.day_counter)

        return ql.RelinkableYieldTermStructureHandle(base_curve)
