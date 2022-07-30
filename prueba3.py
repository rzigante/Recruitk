import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_image():
    uploaded_file = st.file_uploader(label="Sube tu CV en PDF", type=".pdf")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        resume = ""
        for page in reader.pages:
            resume += page.extract_text() + "\n"
        return resume
    else:
        return None


def load_model():
    cv = CountVectorizer()
    return cv


def load_labels():
    job = st.text_area("Pega tu oferta laboral acá",  max_chars = 2000,height=250)
    return job

def predict(job, resume):
   text = [resume,job]
   cv = CountVectorizer()
   count_matrix = cv.fit_transform(text)
   matchpercentage = cosine_similarity(count_matrix)[0][1]
   matchpercentage = round(matchpercentage*100,2)
   #"TU CV calca con la publicación en un " + matchpercentage
   matchpercentage
def main():
    st.title('Hola Recruitk')
    # model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('✨ ¡Compara mi CV!')
    if result:
        st.write('Comparando...')
        predict(categories, image)


if __name__ == '__main__':
    main()