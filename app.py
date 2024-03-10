from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('text_summarization_model.pkl', 'rb') as f:
    text_summarization_model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        print('Getting request')
        text = request.form['input_text']
        print(text)
        summary = text_summarization_model(text)
        return render_template('index.html', summary=summary, text=text)
    return render_template('index.html', summary='Summary of the text will appear here.')

if __name__ == '__main__':
    app.run(debug=True)
