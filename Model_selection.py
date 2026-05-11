import QuantLib as ql
import numpy as np

class InterestRateDataEngine:
    def __init__(self, market_rates, caldates, settle):
        # market_rates: {'1Y': 0.035, '5Y': 0.034, ...}
        self.market_rates = market_rates
        self.day_count = ql.Actual365Fixed()
        self.calendar = caldates
        self.settlement_days = settle
        self.yield_curve = self._build_curve()

    def _build_curve(self):
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        helpers = []
        for tenor, rate in self.market_rates.items():
            years = int(tenor[:-1])
            helpers.append(
                ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)),
                                     ql.Period(years, ql.Years), self.settlement_days,
                                     self.calendar, ql.Following, False, self.day_count))
        curve = ql.PiecewiseLogCubicDiscount(0, self.calendar, helpers, self.day_count)
        curve.enableExtrapolation()
        return curve

    def get_hull_white_input(self, T=5.0, n_steps=500):
        times = np.linspace(0, T, n_steps)
        fwd_rates = [self.yield_curve.forwardRate(t, t, ql.Continuous).rate() for t in times]
        return np.array(fwd_rates, dtype=np.float32)

    def get_lmm_input(self, horizon=5.0, dt=0.25):
        tenors = np.arange(0, horizon + dt, dt)
        initial_forwards = []
        for i in range(len(tenors) - 1):
            fwd = self.yield_curve.forwardRate(tenors[i], tenors[i + 1], ql.Simple, ql.Annual).rate()
            initial_forwards.append(fwd)
        return np.array(initial_forwards, dtype=np.float32)

