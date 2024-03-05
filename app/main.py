import boto3
from io import BytesIO
import pandas as pd
from PIL import Image
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from urllib.parse import urlparse
from .utils.utils import load_env

load_env()

def create_presigned_url(s3_uri: str, expiration: int=3600) -> str:
    s3_client = boto3.client("s3",
                            AWS_ACCESS_KEY_ID=st.secrets["AWS_ACCESS_KEY"],
                            AWS_SECRET_ACCESS_KEY=st.secrets["AWS_SECRET_ACCESS_KEY"],
                            AWS_DEFAULT_REGION='eu-west-1',)
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("Only s3:// URIs are supported.")
    bucket_name, object_name = parsed.hostname, parsed.path.lstrip('/')
    response = s3_client.generate_presigned_url(
        'get_object',
        Params={ 'Bucket': bucket_name, 'Key': object_name },
        ExpiresIn=expiration
    )
    return response

@st.cache_data
def load_csv(s3_url):
    url = create_presigned_url(s3_url)
    #st.write('Loading csv...')
    df = pd.read_csv(url,index_col=0)
    img_names = df.Name
    img_emb = torch.from_numpy(df.iloc[:,1:].values).float()
    return img_names, img_emb

@st.cache_resource
def load_model(model_path):
    #st.write("Loading model...")
    model = SentenceTransformer(model_path,device='cpu')
    return model

img_names, img_emb = load_csv('s3://casa-bugatti/casabugatti-bottles-embeddings.csv')
text_model = load_model('sentence-transformers/clip-ViT-B-32-multilingual-v1')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

def main():
    from . import app_description, app_name
    st.title(app_name)
    st.divider()
    st.write(app_description)
    st.divider()

    number = st.slider('Number of products to show:',1,10,5)
    text_input = st.text_input(
    "Enter query 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="Example: Blue bottle",
    )

    if text_input:
        query_emb = text_model.encode([text_input], convert_to_tensor=True, show_progress_bar=False)
        hits = util.semantic_search(query_emb, img_emb, top_k=number)[0]
        st.subheader("Top products found for you:")

        from itertools import cycle
        cols = cycle(st.columns(3)) 
        image_list = []
        caption = []
        for hit in hits:
            img_url = create_presigned_url(img_names[hit['corpus_id']])
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            image_list.append(img)
            caption.append(img_names[hit['corpus_id']].split('/')[-1].split('.')[0])
        for idx, image_list in enumerate(image_list):
            next(cols).image(image_list, width=300,caption=caption[idx])

if __name__ == '__main__':
    main()