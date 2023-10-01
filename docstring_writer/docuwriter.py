import os
import requests
import logging
class hfAPIModule:
    def __init__(self):
        self.HF_API_KEY=os.environ.get('HF_API_KEY')
        self.HF_MODEL=os.environ.get('HF_MODEL')
        self.headers = {"Authorization": "Bearer " + self.HF_API_KEY}
        self.API_URL = "https://api-inference.huggingface.co/models/"+self.HF_MODEL
    def query(self,payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()
class hfAPIDocuWriter(hfAPIModule):
    def __init__(self):
        super().__init__()
    def add_documentation_to_function(self, text, ):
        output = open(os.path.join(
            os.path.dirname(__file__),'templates', 'function_docstring.txt')
            ,'r').read()
        len_template = len(output)
        output = output.format(user_message=text)
        while "###AI-END-DOC###" not in output[-20:]:
            output = self.query({
                "inputs": output[0]['generated_text'],
            })
        documented_fn_txt = output['generated_text'][len_template:]
        logging.info("documented function output: "+documented_fn_txt)
        return documented_fn_txt


class hfAPIDocuChunker(hfAPIModule):
    def __init__(self):
        super().__init__()
    def make_chunk(self, text):
        output = open(os.path.join(
            os.path.dirname(__file__),'templates', 'chunker_docstring.txt')
            ,'r').read()
        len_template = len(output)
        output = output.format(user_message=text)
        while "###AI-END-DOC###" not in output[-20:]:
            output = self.query({
                "inputs": output[0]['generated_text'],
            })
        chunk = output['generated_text'][len_template:]
        logging.info("documented chunk output: "+chunk)
        return chunk
    def chunk_file(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        chunks = []
        while len(text)>0:
            chunk = self.make_chunk(text)
            chunks.append(chunk)
            text = text[len(chunk):]
        return chunks
