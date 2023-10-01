import unittest
import os
from docstring_writer.docuwriter import  hfAPIDocuChunker
class TestDocumentChunking(unittest.TestCase):
    retries = 0
    def setUp(self) -> None:
        os.environ['HF_API_KEY'] = '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'
        os.environ['HF_MODEL'] = 'Phind/Phind-CodeLlama-34B-v2'
        self.chunker = hfAPIDocuChunker()
    def test_chunking(self):
        file = os.path.dirname(__file__) + '/chunk_test_file.py'
        self.chunker.chunk_file(file)
if __name__ == '__main__':
    unittest.main()
