import unittest
import numpy as np
from numpy.testing import assert_array_equal
import dbbpy

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
            assert_array_equal(false_ixd, np.array([ 99, 124, 143, 145])) 

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

    
if __name__ == '__main__':
    unittest.main()

