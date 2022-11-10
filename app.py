import random

import numpy as np
import streamlit as st
import torch
from omegaconf import OmegaConf
from streamlit_chat import message

import create_instance

conf = OmegaConf.load(f"./config/config-221108-1.yaml")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@st.cache(allow_output_mutation=True)  # 이 annotation을 적어주면 해당 메서드는 처음에만 실행되고 다시 실행되지 않는다.
def load_model():
    dataloader, model = create_instance.new_instance(conf)
    tokenizer = dataloader.tokenizer
    # model = model.load_from_checkpoint(
    #     "save_models/xlm-roberta-large_maxEpoch50_batchSize32_grateful-waterfall-99/epoch=28-step=13543-val_pearson=0.9419651627540588-val_loss=0.35554516315460205.ckpt"
    # )
    model.eval()

    return model, tokenizer


model, tokenizer = load_model()

st.title("Streamlit STS")
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.form(key="my_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([5, 5, 1])

    with col1:
        st.text_input(
            "first sentence",
            placeholder="안녕하세요, 김프로님 식사 하셨나요?",
            key="input1",
            max_chars=512,
        )
    with col2:
        st.text_input(
            "second sentence",
            placeholder="여어, 김프로 식사는 잡쉈어?",
            key="input2",
            max_chars=512,
        )
    with col3:
        st.write("&#9660;&#9660;&#9660;")
        submit = st.form_submit_button(label="Ask")

if submit:
    msg1 = (st.session_state["input1"], True)
    msg2 = (st.session_state["input2"], True)
    if msg1[0] == "":
        msg1 = ("안녕하세요, 김프로님 식사 하셨나요?", True)
    if msg2[0] == "":
        msg2 = ("여어, 김프로 식사는 잡쉈어?", True)

    sep_token = tokenizer.special_tokens_map["sep_token"]
    text = sep_token.join([msg1[0], msg2[0]])
    outputs = tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
    inputs_ids = torch.tensor([outputs["input_ids"]])

    st.session_state.messages.append(msg1)
    st.session_state.messages.append(msg2)
    # st.session_state.messages.append((text.replace(sep_token, ' / '), True))

    for i, msg in enumerate(st.session_state.messages):
        message(msg[0], is_user=msg[1], key=f"{i}")  # key 값을 지정해주지 않으면 같은 문장이 입력으로 들어왔을 때 에러 발생

    with st.spinner("두뇌 풀가동!"):
        result = model(inputs_ids)

    msg = (f"두 문장의 유사도는 {float(result):.4f}입니다.", False)
    st.session_state.messages.append(msg)
    message(msg[0], is_user=msg[1])
