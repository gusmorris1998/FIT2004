from assignment_2 import CatsTrie
import unittest

class Q2Test(unittest.TestCase):
    def test_edge_case_1(self):
        sentences = [
            "fg", "fg", "fg", "fg", "fg", "fg", "abc", "abc", "abc", "abc", "abc",
            "abdc", "abdc", "abdc", "abdz", "abdz", "abdz", "abdz", "ac", "ac", "ac", "ac", "ac"
        ]
        mycattrie = CatsTrie(sentences)
        self.assertEqual(mycattrie.autoComplete(""), "fg")
        self.assertEqual(mycattrie.autoComplete("z"), None)
        self.assertEqual(mycattrie.autoComplete("a"), "abc")
        self.assertEqual(mycattrie.autoComplete("ab"), "abc")
        self.assertEqual(mycattrie.autoComplete("abd"), "abdz")
        self.assertEqual(mycattrie.autoComplete("abdd"), None)
        self.assertEqual(mycattrie.autoComplete("bdz"), None)
        self.assertEqual(mycattrie.autoComplete("dz"), None)
        self.assertEqual(mycattrie.autoComplete("bc"), None)

    def test_edge_case_2(self):
        sentences = [
            "f", "f", "f", "f", "f", "f", "abca", "abca", "abca", "abca",
            "abc", "abc", "abc", "abc", "ad", "ad", "ad", "ad", "ad"
        ]
        mycattrie = CatsTrie(sentences)
        self.assertEqual(mycattrie.autoComplete(""), "f")
        self.assertEqual(mycattrie.autoComplete("a"), "ad")
        self.assertEqual(mycattrie.autoComplete("abc"), "abc")
        self.assertEqual(mycattrie.autoComplete("abcd"), None)

if __name__ == "__main__":
    unittest.main()