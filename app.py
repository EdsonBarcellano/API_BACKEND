from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').lower()
    name = data.get('name', 'User')  # Default to "User" if name not provided

    if message == 'hi':
        return jsonify({'reply': f'Hello, Edson Barcellano!'})
    else:
        return jsonify({'reply': f'You said: {message}'})

if __name__ == '__main__':
    app.run(debug=True)
