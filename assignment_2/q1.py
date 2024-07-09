import unittest
__author__ = "Angus Corr"
from problem_1 import maxThroughput

class Q1Test(unittest.TestCase):
    
    def test_provided(self):
        """
        Using test provided by problem sheet.
        """
        connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),
        (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
        maxIn = [5000, 3000, 3000, 3000, 2000]
        maxOut = [5000, 3000, 3000, 2500, 1500]
        origin = 0
        targets = [4, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 4500)

    def test_0(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
        (2, 3, 50)]
        maxIn = [1, 60, 50, 7]
        maxOut = [1, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 1)

    def test_1(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
        (2, 3, 50)]
        maxIn = [20, 60, 50, 7]
        maxOut = [20, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 7)
    
    def test_2(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
        (2, 3, 50)]
        maxIn = [1, 60, 50, 7]
        maxOut = [20, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 7)

    def test_3(self):
        connections = [(0, 1, 20), (1, 2, 20)]
        maxIn = [20, 5, 20]
        maxOut = [20, 100, 20]
        origin = 0
        targets = [1]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 5)

    def test_4(self):
        connections = [(0, 1, 50), (1, 2, 20), (1, 3, 35)]
        maxIn = [60, 60, 10, 30]
        maxOut = [60, 50, 30, 2]
        origin = 0
        targets = [2, 3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 40)

if __name__ == "__main__":
    unittest.main()