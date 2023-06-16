from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Carga el modelo de clasificación de texto previamente entrenado
model = tf.keras.models.load_model('miModelo_texto_clasificacion')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    
    # Realiza la clasificación utilizando el modelo cargado
    prediction = model.predict([text])
    
    # Determina la clase predicha (ajústalo según tu modelo)
    if prediction[0][0] > 0:
        result = 'Crítica positiva'
    else:
        result = 'Crítica negativa'
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
