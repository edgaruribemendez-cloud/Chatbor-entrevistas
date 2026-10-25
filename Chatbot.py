import json
import random
import nltk
import pathlib

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# 1. Cargar el dataset de intenciones (intents.json)
rutaBase =pathlib.Path(__file__).resolve()
print(rutaBase.parent) # Imprime el directorio de trabajo actual para verificar la ruta
with open(rutaBase.parent / 'intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Función clave utilizando NLTK para limpiar y normalizar el texto
def limpiar_y_procesar(texto):
    # Tokenizar (dividir en palabras) y pasar a minúsculas
    tokens = nltk.word_tokenize(texto.lower())
    # Lematizar (reducir la palabra a su raíz, ej: "corriendo" -> "correr")
    tokensLimpios = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    #print(tokensLimpios)  # Imprime los tokens limpios para depuración
    return " ".join(tokensLimpios)

# 2. Preparar los datos de entrenamiento
texts = []
labels = []
respuestaDict = {}

for intent in data['intents']:
    tag = intent['tag']
    respuestaDict[tag] = intent['responses']
    for pattern in intent['patterns']:
        # Aplicamos el procesamiento de NLTK a cada patrón de entrenamiento
        texto_procesado = limpiar_y_procesar(pattern)
        texts.append(texto_procesado)
        labels.append(tag)

# 3. Vectorización y Modelo de Machine Learning
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)


# 4. Bucle principal de la entrevista
def iniciar_chat():
    print("Bot: ¡Hola! Bienvenido al simulador de entrevistas. ¿Cómo te llamas y qué puesto buscas?")
    
    while True:
        user_input = input("Tú: ").strip()
        
        if user_input.lower() in ["salir", "exit", "terminar"]:
            print("\nBot: ¡Mucho éxito en tu proceso de selección real! Hasta luego.")
            break
            
        if not user_input:
            continue

        # Procesar la entrada del usuario usando NLTK
        input_procesado = limpiar_y_procesar(user_input)
        
        # Vectorizar y predecir la intención
        X_test = vectorizador.transform([input_procesado])
        prediccion = model.predict(X_test)[0]
        
        # Seleccionar una respuesta aleatoria de la intención detectada
        respuesta = random.choice(respuestaDict[prediccion])
        print(f"\nBot: {respuesta}\n")

if __name__ == "__main__":
    iniciar_chat()