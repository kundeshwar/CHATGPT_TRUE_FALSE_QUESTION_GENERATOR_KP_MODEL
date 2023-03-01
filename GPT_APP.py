#-------------------------------------------
#import necessary labrary
import pickle
import torch
import pickle
import requests
#-------------------------first import our pickle file for model 
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
model = pickle.load(open('model_1.pkl', "rb"))
tokenizer = pickle.load(open("model_2.pkl", "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#---------------------------------------------to generate question
def greedy_decoding (inp_ids,attn_mask):
    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
        return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding (inp_ids,attn_mask):
        topkp_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               do_sample=True,
                               top_k=40,
                               top_p=0.80,
                               num_return_sequences=3,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                               )
        Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
        return [Question.strip().capitalize() for Question in Questions]

#--------------------------------------------
#import useful labrary
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

st.set_page_config(page_title="TRUE OR FALSE QUESTION GENERATER GPT (KP MODEL)", page_icon=":sunglasses:", layout="centered", initial_sidebar_state="expanded")
st.title("""TRUE OR FALSE QUESTION GENERATER GPT (KP MODEL)""")
#--------------------------------------------

st.markdown("""  
<style>
.css-1rs6os.edgvbvh3
{
    visibility: hidden;
}
</style>""",unsafe_allow_html=True)
#following code is used to remove "made with streamlit" 
st.markdown("""
<style>
.css-1lsmgbg.egzxvld0
{
    visibility: hidden;
 } 
</style>""", unsafe_allow_html=True)
#---------------------------------------------title of web app
#st.set_page_config(page_title="TRUE OR FALSE QUESTION GENERATER GPT (KP MODEL)", page_icon=":sunglasses:", layout="centered", initial_sidebar_state="expanded")
#------------------------------------------for sidebar
st.sidebar.markdown(f"<h3 style='text-align: center;'>KP CHAT GPT</h3>",unsafe_allow_html=True)
st.sidebar.markdown("This is KP CHAT GPT you can use for generate True or flase question. It will Automatically Generate True or False questions from any content")
#------------------------------------------for lottie animation 
def lottie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
loettir_1 = lottie("https://assets10.lottiefiles.com/packages/lf20_7fCbvNSmFD.json")
loettir_2 = lottie("https://assets2.lottiefiles.com/packages/lf20_qavaymcn.json")
loettir_3 = lottie("https://assets3.lottiefiles.com/packages/lf20_CZva9peGiW.json")
loettir_4 = lottie("https://assets6.lottiefiles.com/packages/lf20_BgywoUBeiL.json")

#------------------------------------------------
st.markdown(f"<h6 style='text-align: center;'>Please Upload Your Context</h6>",unsafe_allow_html=True)

passage = st.text_area("Paste Here")
option = st.selectbox(
    'SELECT QUESTION TYPE',
    ('True', 'False'))
submit = st.button("SEE QUESTION")
st_lottie(loettir_2)
if submit and option=="True":
        truefalse ="yes"
        text = "truefalse: %s passage: %s </s>" % (truefalse, passage)
        max_len = 256

        encoding = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        output = beam_search_decoding(input_ids,attention_masks)
        st.write("More Accurate:- ")
        for out in output:
            st.write(out)
        st_lottie(loettir_4)

        output = topkp_decoding(input_ids,attention_masks)
        st.write("Accuracy is less But New questions:- ")
        for out in output:
            st.write(out)
        st_lottie(loettir_3)
elif submit and option=="False":
        truefalse ="no"
        text = "truefalse: %s passage: %s </s>" % (truefalse, passage)
        max_len = 256

        encoding = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        output = beam_search_decoding(input_ids,attention_masks)
        st.write("Accuracy is more:- ")
        for out in output:
            st.write(out)
        st_lottie(loettir_1)

        output = topkp_decoding(input_ids,attention_masks)
        st.write("Accuracy is less But New questions:- ")
        for out in output:
            st.write(out)
        st_lottie(loettir_4)
