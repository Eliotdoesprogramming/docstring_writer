import os
import time
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
        self.template = open(os.path.join(
            os.path.dirname(__file__),'templates', 'function_docstring.txt')
            ,'r').read()
    def add_documentation_to_function(self, text, ):
        output = self.template.format(user_message=text)
        len_template = len(output)
        output = [{"generated_text": output}]

        while "###AI-END-DOC###" not in output[-10:]:
            output = self.query({
                "inputs": output[0]['generated_text'],
            })
        documented_fn_txt = output[0]['generated_text'][len_template:]
        logging.info("documented function output: "+documented_fn_txt)
        return documented_fn_txt


class hfAPIDocuChunker(hfAPIModule):
    retries = 0
    def __init__(self):
        super().__init__()
        self.template = open(os.path.join(
            os.path.dirname(__file__),'templates', 'file_chunker3.txt')
            ,'r').read()
    def make_chunk(self, text):
        output = self.template.format(user_message=text)
        len_template = len(output)
        prev_len = 0
        output = [{"generated_text": output}]
        try:
            while len(output[0]['generated_text']) != prev_len:
                prev_len = len(output[0]['generated_text'])
                output = self.query({
                    "inputs": output[0]['generated_text'],
                    "wait_for_model":True
                })
        except Exception as e:
            logging.error(e)
            logging.error(output)
            if 'estimated_time' in output.keys():
                self.retries +=1
                if self.retries < 3:
                    time.sleep(output['estimated_time'])
                    self.make_chunk(text)
                else:
                    raise e
            else:
                raise e
        chunk = output[0]['generated_text'][len_template:]
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
