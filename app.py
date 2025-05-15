import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.title("🧠 Mental Health Chatbot")
st.write("I'm here to listen. You can talk to me anytime 💙")

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {response}")
    st.session_state.user_input = ""
