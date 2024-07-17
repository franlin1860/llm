from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 设置PDF解析器
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./pdf_data", file_extractor=file_extractor
).load_data()

# 将解析后的文档保存到文件
with open("./data/parsed_documents.txt", "w") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")
