import dbbpy
import unittest
import numpy as np
from numpy.testing import assert_array_equal

class Testdbbpy(unittest.TestCase):
    
    def setUp(self):

        self.fw_data = np.loadtxt("tests/forward1.txt", delimiter='\t')
        data = np.loadtxt("tests/spopt1.txt", delimiter='\t')
        
        self.strike_list = data[:, 0]
        self.pFlag_list = data[:, 1]
        self.bid_list = data[:, 2]
        self.ask_list = data[:, 3]

        min_strike = int(np.min(self.strike_list))
        max_strike = int(np.max(self.strike_list))

        sp = list(range(0, min_strike, 10)) + list(range(min_strike, max_strike, 5)) + list(range(max_strike, 4440, 10))
        
        self.sp_np = np.array(sp)
        self.bid_np = np.array(self.bid_list)
        self.ask_np = np.array(self.ask_list)
        self.strike_np = np.array(self.strike_list)
        self.pFlag_np = np.array(self.pFlag_list, dtype=bool)
        self.fw = self.fw_data[0]

    
    def test_getFeasibleOptionFlags(self):
        
        feasible_options = dbbpy.getFeasibleOptionFlags(self.sp_np, self.bid_np, self.ask_np, self.strike_np, self.pFlag_np, self.fw, self.fw-0.02, self.fw+0.02)
        false_ixd = np.where(feasible_options == False)
        
        with self.assertRaises(AssertionError):
            assert_array_equal(false_ixd, np.array([99, 124, 143, 145])) 

    def test_getMidPriceQ(self):

        feasible_options = dbbpy.getFeasibleOptionFlags(self.sp_np, self.bid_np, self.ask_np, self.strike_np, self.pFlag_np, self.fw, self.fw-0.02, self.fw+0.02)
        q = dbbpy.getMidPriceQ(
            self.sp_np, self.bid_np[feasible_options], self.ask_np[feasible_options], 
            self.strike_np[feasible_options], self.pFlag_np[feasible_options],  self.fw, self.fw-0.02, self.fw+0.02
            )

        self.assertAlmostEqual(np.sum(q), 1)
        self.assertLess(np.abs(q.dot(self.sp_np) - self.fw), 0.02)
        
    
    def test_getMidPriceQReg(self):

        feasible_options = dbbpy.getFeasibleOptionFlags(self.sp_np, self.bid_np, self.ask_np, self.strike_np, self.pFlag_np, self.fw, self.fw-0.02, self.fw+0.02)
        q = dbbpy.getMidPriceQReg(
            self.sp_np, self.bid_np[feasible_options], self.ask_np[feasible_options], 
            self.strike_np[feasible_options], self.pFlag_np[feasible_options],  self.fw, self.fw-0.02, self.fw+0.02
            )

        self.assertAlmostEqual(np.sum(q), 1)
        self.assertLess(np.abs(q.dot(self.sp_np) - self.fw), 0.02)


    def test_getQReg(self):

        feasible_options = dbbpy.getFeasibleOptionFlags(self.sp_np, self.bid_np, self.ask_np, self.strike_np, self.pFlag_np, self.fw, self.fw-0.02, self.fw+0.02)
        q = dbbpy.getQReg(
            self.sp_np, self.bid_np[feasible_options], self.ask_np[feasible_options], 
            self.strike_np[feasible_options], self.pFlag_np[feasible_options],  self.fw, self.fw-0.02, self.fw+0.02
            )

        self.assertAlmostEqual(np.sum(q), 1)
        self.assertLess(np.abs(q.dot(self.sp_np) - self.fw), 0.02)

    def test_optimize(self):
        n = 4
        alpha = 1.3
        lambda_ = 1.1
        omega_l = np.array([0, 3])
        sp = np.array([1200, 1250, 1300, 1350])
        strike = np.array([1290, 1295, 1295, 1300])
        bid = np.array([27.7, 27.4, 29.4, 25.0])
        ask = np.array([29.3, 29.7, 31.4, 26.9])
        pFlag = np.array([1,0,1,0])

        result = dbbpy.performOptimization(n, alpha, lambda_, omega_l, sp, strike, bid, ask, pFlag)
        p = result[0]
        q = result[1]

        exp_p = np.array([0.429653, 0.0304595, 0.0442842, 0.495604])
        exp_q = np.array([0.325555, 2.34516e-07, 0.147889, 0.526556])

        for i in range(len(p)):
            self.assertAlmostEqual(p[i], exp_p[i], 5) 
            self.assertAlmostEqual(q[i], exp_q[i], 5) 

    
if __name__ == '__main__':
    unittest.main()

