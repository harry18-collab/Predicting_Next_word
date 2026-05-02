import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_resources():
    model = load_model('predict_nextword_LSTM.h5')
    with open('tokenizer_LSTM', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_next_word(model, tokenizer, text):
    max_seq_len = model.input_shape[1]
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = [min(t, len(tokenizer.word_index) - 1) for t in token_list]

    if len(token_list) >= max_seq_len:
        token_list = token_list[-max_seq_len:]

    token_list = pad_sequences([token_list],
                                maxlen=max_seq_len,
                                padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# ── UI ────────────────────────────────────────────────────
st.title("Next Word Predictor")
st.subheader("Trained on Shakespeare's Hamlet")

model, tokenizer = load_resources()

input_text = st.text_input("Enter your text:",
                            placeholder="e.g. to be or not to")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict Next Word", use_container_width=True):
        if input_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            predicted = predict_next_word(model, tokenizer, input_text)
            st.success(f"**Next word:** `{predicted}`")
            st.write(f"**Full:** {input_text} **{predicted}**")

with col2:
    if st.button("Predict 5 Words", use_container_width=True):
        if input_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            sentence = input_text
            for _ in range(5):
                next_word = predict_next_word(model, tokenizer, sentence)
                if next_word:
                    sentence += " " + next_word
            st.success(f"**Extended:** `{sentence}`")