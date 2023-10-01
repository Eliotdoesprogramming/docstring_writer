import os
import requests
from docstring_writer import template_path
import logging
class hfAPIDocuWriter:
    def __init__(self):
        self.HF_API_KEY=os.environ.get('HF_API_KEY')
        self.HF_MODEL=os.environ.get('HF_MODEL')
        self.headers = {"Authorization": "Bearer " + self.HF_API_KEY}
        self.API_URL = "https://api-inference.huggingface.co/models/"+self.HF_MODEL
    def add_documentation_to_function(self, text, ):
        output = open(os.path.join(template_path, 'function_docstring.txt','r')).read()
        len_template = len(output)
        output = output.format(user_message=text)
        while "###AI-END-DOC###" not in output[-20:]:
            output = self.query({
                "inputs": output[0]['generated_text'],
            })
        documented_fn_txt = output['generated_text'][len_template:]
        logging.info("documented function output: "+documented_fn_txt)
        return documented_fn_txt
    def query(self,payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()
