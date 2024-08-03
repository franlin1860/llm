from flask import Flask, render_template, jsonify, request
from pg_schema_extractor import PGSchemaExtractor

app = Flask(__name__)
extractor = PGSchemaExtractor()


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

        extractor.set_schema(schema_type)

        # 打印调试信息
        print("In extract()\n")

        index = extractor.extract(data_dir)

        response = extractor.query(index, query_text)

        return jsonify({"message": str(response)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
