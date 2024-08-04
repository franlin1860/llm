from flask import Flask, render_template, jsonify, request
import subprocess
import json
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    try:
        data = request.json
        schema_type = data['schema']
        query_text = data['query']

        data_dir = "./data/paul_graham/"  # 确保这个目录存在并包含所需的数据

        print(f"Schema type: {schema_type}\n")

        # 准备参数并调用外部脚本
        script_path = os.path.join(os.path.dirname(__file__), 'pg_schema_extractor.py')
        args = json.dumps({"schema": schema_type, "query": query_text, "data_dir": data_dir})

        result = subprocess.run(['python3', script_path, args], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        
        response = json.loads(result.stdout)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
