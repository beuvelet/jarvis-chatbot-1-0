import streamlit as st
from system_cuda import config
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv
import os
import torch
import datetime

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer l'heure actuelle
heure_actuelle = datetime.datetime.now().strftime("%H:%M:%S")

# Récupérer la date actuelle
date_actuelle = datetime.datetime.now().strftime("%Y-%m-%d")

with open(".streamlit/prompt_sys.txt", "r") as file:
    Prompt_system = file.read()

# Utilisation de la configuration du noyau CUDA
num_cuda_cores = config.NUM_CUDA_CORES
print(f"Nombre de noyaux CUDA : {num_cuda_cores}")

# Vérifiez si CUDA est disponible
if torch.cuda.is_available():
    # Initialisez CUDA
    torch.cuda.init()
    # Utilisez le premier GPU disponible
    device = torch.device("cuda")
    print("GPU est disponible et sera utilisé.")
else:
    # Si CUDA n'est pas disponible, utilisez le CPU
    device = torch.device("cpu")
    print("CUDA n'est pas disponible. Le CPU sera utilisé.")

# Utilisez le GPU pour les calculs si CUDA est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Créez un tenseur PyTorch et transférez-le sur le GPU
tensor = torch.tensor([1, 2, 3]).to(device)

# Charger les modèles
Mistral_7b_Q5 = os.getenv("MISTRAL_MODEL_7B_Q5")
Mistral_Instruct_7B_Q5 = os.getenv("MISTRAL_MODEL_INSTRUCT_7B_Q5")

# Définir les paramètres du modèle
n_gpu_layers = 1
n_ctx = 4096
n_batch = 2048

# Activer Wide mode
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title('💬 Jarvis Chatbot 1.0')

# Modèles et paramètres
selected_model = st.sidebar.selectbox('Choisissez le modèle',
                                      ['Mistral_7b_Q5', 'Mistral_Instruct_7B_Q5', 'Mistral_7b_Q5', 'Mistral_7b_Q6',
                                       'Mistral_7b_Q7'],
                                      key='selected_model')
temperature = st.sidebar.slider('Température', min_value=0.01, max_value=5.0, value=0.7, step=0.01)
top_p = st.sidebar.slider('probabilité maximale', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('Longueur maximale', min_value=128, max_value=4096, value=512, step=1)

# Charger le modèle sur le GPU si disponible, sinon sur le CPU
llm = LlamaCpp(model_path=Mistral_7b_Q5 if torch.cuda.is_available() else Mistral_Instruct_7B_Q5,
               n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_batch=n_batch, top_p=top_p, max_length=max_length,
               repetition_penalty=1, f16_kv=False, verbose=True, stream=True)

# Stocker les réponses générées par LLM
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Comment puis-je vous aider aujourd'hui ?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Comment puis-je vous aider aujourd'hui ?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):

    string_dialogue = heure_actuelle + date_actuelle + Prompt_system

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = llm.invoke(f"{string_dialogue} {prompt_input} Assistant:")
    return output


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
