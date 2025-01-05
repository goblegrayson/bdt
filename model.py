import numpy as np
import pandas as pd
from scipy import optimize


class BlackDermanToy:
    def __init__(self, term_struct):
        self.observed_term_struct = term_struct
        self.dt = 1  # Years

    @property
    def observed_term_struct(self):
        return self._observed_term_struct

    @observed_term_struct.setter
    def observed_term_struct(self, value):
        # Set the value
        self._observed_term_struct = value.sort_values('mat')
        self.rate_tree = None
        self.dts = None
        self.debug = True
        # Any time the underlying observation changes, update the refit the model
        self.update()

    @observed_term_struct.deleter
    def observed_term_struct(self):
        del self._observed_term_struct

    @staticmethod
    def calc_implied_vol(ru, rd):
        return 0.5 * np.log(ru / rd)

    @staticmethod
    def calc_discounted_price(price, dt, r):
        return price / ((1 + r) ** dt)

    @staticmethod
    def calc_yield(price_future, price_current, dt):
        return (price_future / price_current) ** (1 / dt) - 1

    def build_trees(self, r_vec, rate_tree, dts, par):
        # Find the first empty step
        rate_tree = rate_tree.copy()
        i_solve = int(np.searchsorted(rate_tree[0, :] == 0, 1))
        # Determine the boundary node of the tree
        i_high = 0
        i_low = i_solve
        # Solve for the rates
        rate_tree[i_high, i_solve] = r_vec[0]
        rate_tree[i_low, i_solve] = r_vec[1]
        for i_mid in range(i_high + 1, i_low):
            rate_vol = 0.5 * np.log(r_vec[0] / r_vec[1]) / i_low
            rate_tree[i_mid, i_solve] = rate_tree[i_mid-1, i_solve] / np.exp(2 * rate_vol)
        # Solve for the prices and yields
        price_tree = np.zeros((i_solve + 2, i_solve + 2))
        price_tree[:, i_solve + 1] = par
        yield_tree = np.zeros((i_solve + 1, i_solve + 1))
        for i_step in range(i_solve, -1, -1):
            dt = dts[i_step]
            for i_node in range(i_step + 1):
                # Price
                price_up = price_tree[i_node, i_step + 1]
                price_down = price_tree[i_node + 1, i_step + 1]
                price_exp = 0.5 * (price_up + price_down)
                r_local = rate_tree[i_node, i_step]
                price_tree[i_node, i_step] = self.calc_discounted_price(price_exp, dt, r_local)
                # Yield
                time_to_maturity = np.sum(dts[i_step:i_solve + 1])
                yield_tree[i_node, i_step] = self.calc_yield(par, price_tree[i_node, i_step], time_to_maturity)
        # Output the resulting trees
        return rate_tree, price_tree, yield_tree

    def eval_trees(self, r_vec, rate_tree, dts, par, yld_target=0, vol_target=0):
        rate_tree, price_tree, yield_tree = self.build_trees(r_vec, rate_tree, dts, par)
        yld0 = yield_tree[0, 0]
        vol0 = 0.5 * (np.log(yield_tree[0, 1] / yield_tree[1, 1]))
        return yld0-yld_target, vol0-vol_target

    def update(self):
        # Build the new short-term rate tree
        term_struct = self.observed_term_struct
        n_steps = len(term_struct)
        rate_tree = np.zeros((n_steps, n_steps))
        self.dts = np.diff(term_struct['mat'])
        self.dts = np.append(self.dts, 1)
        # For the one-year zero, the yield is the implied rate by definition
        r0 = term_struct.iloc[0]['yld']
        rate_tree[0, 0] = r0
        # For the multi-year zero, the implied rate has to be numerically solved, backing today's price out through the tree
        for i_step in range(1, n_steps):
            # Find target params
            yld_target = term_struct.iloc[i_step]['yld']
            vol_target = term_struct.iloc[i_step]['vol']
            # Set up cost function
            ru_guess = .2179
            rd_guess = .0872
            x0 = [ru_guess, rd_guess]
            # Solve the step
            ru, rd = optimize.fsolve(self.eval_trees, x0, args=(rate_tree, self.dts, term_struct.iloc[i_step]['par'], yld_target, vol_target))
            if i_step == n_steps or self.debug:
                rate_tree, price_tree, yield_tree = self.build_trees((ru, rd), rate_tree, self.dts, term_struct.iloc[i_step]['par'])
            else:
                rate_tree, _, _ = self.build_trees((ru, rd), rate_tree, self.dts, term_struct.iloc[i_step]['par'])
        # Save the rate tree
        self.rate_tree = rate_tree

    def quote_zero(self, maturity, par):
        # Down-select the rate tree to just the nodes we need
        rate_tree = self.rate_tree.copy()
        i_solve = int(np.searchsorted(np.cumsum(self.dts), maturity))
        # Solve for the prices and yields
        price_tree = np.zeros((i_solve + 2, i_solve + 2))
        price_tree[:, i_solve + 1] = par
        for i_step in range(i_solve, -1, -1):
            dt = self.dts[i_step]
            for i_node in range(i_step + 1):
                price_up = price_tree[i_node, i_step + 1]
                price_down = price_tree[i_node + 1, i_step + 1]
                price_exp = 0.5 * (price_up + price_down)
                r_local = rate_tree[i_node, i_step]
                price_tree[i_node, i_step] = self.calc_discounted_price(price_exp, dt, r_local)
        return price_tree[0, 0]

    def quote_tbond(self, maturity, par, coupon, coupons_per_year=2):
        # Decompose the tbond into a zero for the principle maturity and final coupon plus a zero for all other coupons sum the prices of each.
        price = self.quote_zero(maturity, par + par * coupon)
        dt_coupon = 1 / coupons_per_year
        for i_coupon in range(maturity * coupons_per_year - 1):
            coupon_maturity = dt_coupon + dt_coupon * i_coupon
            price += self.quote_zero(coupon_maturity, par * coupon)
        return price


if __name__ == '__main__':
    # Create a test term structure
    term_struct = pd.DataFrame(
        {
            'mat': np.arange(1, 6),
            'yld': np.array([.10, .11, .12, .125, .13]),
            'vol': np.array([.20, .19, .18, .17, .16]),
            'par': np.array([100, 100, 100, 100, 100])
        }
    )
    # Instantiate and calibrate model
    BDT = BlackDermanToy(term_struct)
    # Run test case: rate tree from the original paper
    expected_rate_tree = np.array([
        [0.1000, 0.1432, 0.1942, 0.2179, 0.2553],
        [0.0000, 0.0979, 0.1377, 0.1606, 0.1948],
        [0.0000, 0.0000, 0.0976, 0.1183, 0.1486],
        [0.0000, 0.0000, 0.0000, 0.0872, 0.1134],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0865]
    ])
    assert np.isclose(BDT.rate_tree, expected_rate_tree, 1e-3).all()
    # Now use the tbond from figure G of the original paper to ensure we can quote bond prices
    expected_bond_price = 95.51
    calculated_bond_price = BDT.quote_tbond(3, 100, 0.1, 1)
    assert np.isclose(expected_bond_price, calculated_bond_price, 1e-2)  # Bit of round-off error here I think

