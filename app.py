import streamlit as st
from image_search import load_model, process_image, process_text, search_images

st.set_page_config(
        page_title="Bangla CLIP Search",
        page_icon="chart_with_upwards_trend"
    )
st.markdown(
    """
<style>
#introduction {
    padding: 10px 20px 10px 20px;
    background-color: #aad9fe;
    border-radius: 10px;

}

#introduction p {
    font-size: 1.1rem;
    color: #050e14;

}

img {
    padding: 5px;
}
</style>


""",
    unsafe_allow_html=True,
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.markdown("# বাংলা CLIP সার্চ ইঞ্জিন ")
st.markdown("""---""")
st.markdown(
    """
<div id="introduction">

<p>
Contrastive Language-Image Pre-training (CLIP), consisting of a simplified version of ConVIRT trained from scratch, is an efficient method of image representation learning from natural language supervision. , CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. 

The model consists of an EfficientNet image encoder and a BERT encoder and was trained on multiple datasets from Bangla image-text domain.

</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("""---""") 
text_query = st.text_input(":mag_right: Search Images / ছবি খুজুন", "সুন্দরবনের নদীর পাশে একটি বাঘ")
st.markdown("""---""") 
number_of_results = st.slider("Number of results ", 1, 100, 10)
st.markdown("""---""") 

ret_imgs, ret_scores, _, _ = search_images(text_query, "demo_images/", k = number_of_results)

st.markdown("<div style='align: center; display: flex'>", unsafe_allow_html=True)
st.image([str(result) for result in ret_imgs], caption = ["Score: " + str(r_s) for r_s in ret_scores], width=230)
st.markdown("</div>", unsafe_allow_html=True)