import unittest
import os
from docstring_writer.docuwriter import  hfAPIDocuChunker
class TestDocumentChunking(unittest.TestCase):
    retries = 0
    def setUp(self) -> None:
        self.chunker = hfAPIDocuChunker()
    def test_chunking(self):
        file = os.path.dirname(__file__) + '/chunk_test_file.py'
        self.chunker.chunk_file(file)
if __name__ == '__main__':
    unittest.main()
