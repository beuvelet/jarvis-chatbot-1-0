import os
import langchain as lc
from langchain_community.llms import LlamaCpp
import time
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Configuration du benchmark
model_names = ["MISTRAL_MODEL_7B_Q5", "MISTRAL_MODEL_INSTRUCT_7B_Q5"]
num_iterations = 10  # Nombre d'itérations à exécuter pour chaque modèle


# Fonction pour charger et exécuter un modèle GGUF
def run_model(model_name):
    model_path = os.getenv(model_name)
    if model_path is None:
        print(f"Erreur : Impossible de trouver le chemin vers le modèle {model_name} dans le fichier .env")
        return
    # Charger le modèle avec LlamaCpp
    llm = LlamaCpp(model_path=model_path)

    # Assurez-vous que input_data est une chaîne de caractères valide
    input_data = " Utilisez une phrase d'exemple pertinente pour votre domaine d'application. Par exemple, si vous utilisez un modèle de génération de texte pour la rédaction automatique d'articles, vous pourriez utiliser une phrase d'exemple qui résume le sujet de l'article."
    start_time = time.time()
    for _ in range(num_iterations):
        output = llm.predict(input_data)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    print(f"Modèle: {model_name}, Temps moyen par itération: {avg_time:.4f} secondes")


# Exécution du benchmark pour chaque modèle
for model_name in model_names:
    run_model(model_name)
