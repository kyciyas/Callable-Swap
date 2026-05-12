import numpy as np
from arch import arch_model
import warnings

# 수치 연산 경고 무시
warnings.filterwarnings("ignore")


class VolatilityEngine:
    def __init__(self, data, annualize_factor: int = 252):
        arr = np.array(data, dtype=np.float64)
        arr = arr[arr > 0]

        if len(arr) < 2:
            self.returns = np.array([0.0] * 10)
        else:
            self.returns = np.diff(np.log(arr)) * 100

        self.annualize_factor = annualize_factor

    def get_ewma_vol(self, lam: float = 0.94):
        if len(self.returns) < 2 or np.std(self.returns) == 0:
            return 0.20

        n = len(self.returns)

        weights = (1 - lam) * (lam ** np.arange(n)[::-1])
        variance = np.sum(weights * (self.returns ** 2)) / np.sum(weights)

        daily_vol = np.sqrt(variance)

        return (daily_vol / 100) * np.sqrt(self.annualize_factor)

    def get_garch_vol(self):
        if len(self.returns) < 30:
            return self.get_ewma_vol(), None

        try:
            model = arch_model(self.returns, p=1, q=1, vol='Garch', dist='normal', rescale=True)
            res = model.fit(disp='off')

            scale = res.scale if hasattr(res, 'scale') else 1.0
            current_daily_vol = res.conditional_volatility[-1] / scale

            annual_vol = (current_daily_vol / 100) * np.sqrt(self.annualize_factor)

            return annual_vol, res.params
        except:
            return self.get_ewma_vol(), None

    def get_comparison_report(self):
        ewma_val = self.get_ewma_vol()
        garch_val, _ = self.get_garch_vol()

        if np.isnan(garch_val) or garch_val <= 0:
            garch_val = ewma_val

        print(f"\n[Volatility Analysis] Target: Market Time-series")
        print(f" - EWMA Vol: {ewma_val:.2%}")
        print(f" - GARCH Vol: {garch_val:.2%}")

        return {"ewma": ewma_val, "garch": garch_val}
