import numpy as np
import model # 自動會去載我們的程式
from flask import Flask, request, jsonify
from flask_cors import CORS
# CORS: 跨來源的支援共享，讓我們的網站可以存取不同、跨網域的伺服器支援

app = Flask(__name__)
CORS(app) # 讓大家可以存取我們的app

@app.route('/') # 路由、進入點
def index():
    return 'hello!!' # 進入路由，就自動回傳字串

@app.route('/predict', methods=['POST'])
def postInput():
    insertValues = request.get_json()
    x = insertValues['description']
    input = np.array([[x,]])
    process_data = model.trans(input)
    result = model.predict(process_data)
    return jsonify({'return': str(result)})

# 建立port號
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) # 當api的入口點，有debug，更新code的時候就不用自己shut down再重run
