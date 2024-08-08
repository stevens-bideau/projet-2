import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import base64

# Charger les données depuis l'URL
url = 'https://www.phipix.com/data_projet2/df_final_ml.csv'
df_ml_reco = pd.read_csv(url)

# Réorganiser les colonnes pour que 'title' apparaisse en deuxième position après 'tconst'
cols = ['tconst', 'title']
for col in df_ml_reco.columns:
    if col not in ['tconst', 'title']:
        cols.append(col)
df_ml_reco = df_ml_reco[cols]

# Remplacer la release_date sous forme de date par l'année
df_ml_reco['year'] = pd.to_datetime(df_ml_reco['release_date'], errors='coerce').dt.year.astype('Int64')

# Supprimer les valeurs manquantes notamment dans la colonne 'year'
df_ml_reco = df_ml_reco.dropna(subset=['year']).reset_index(drop=True)

# Extraction et transformation des titres de films en utilisant TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_ml_reco['title'])

# Normalisation des autres caractéristiques en excluant la colonne 'year'
numeric_cols = df_ml_reco.select_dtypes(include=['int64', 'float64']).columns.drop('year')
features = df_ml_reco[numeric_cols]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Combinaison des deux ensembles de caractéristiques
tfidf_matrix_array = tfidf_matrix.toarray()
if tfidf_matrix_array.shape[0] != scaled_features.shape[0]:
    raise ValueError("Les dimensions de tfidf_matrix et scaled_features ne correspondent pas.")

final_features = np.concatenate((tfidf_matrix_array, scaled_features), axis=1)

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
final_features = imputer.fit_transform(final_features)

# Utilisation de KNN pour la recherche des voisins les plus proches
knn = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='cosine')
knn.fit(final_features)

def find_similar_movies(movie_title, knn, df, final_features, n_neighbors=10):
    if movie_title not in df['title'].values:
        st.write(f"Le film '{movie_title}' n'a pas été trouvé dans le DataFrame.")
        return None

    movie_index = df[df['title'] == movie_title].index[0]
    movie_vector = final_features[movie_index].reshape(1, -1)

    distances, indices = knn.kneighbors(movie_vector, n_neighbors=n_neighbors)
    
    similar_movies = df.iloc[indices[0]].copy()
    similar_movies['distance'] = distances[0]
    return similar_movies

# Fonction pour afficher des films avec un bouton pour les détails
def display_movies(movies):
    num_columns_per_row = 5
    num_movies = len(movies)
    num_rows = (num_movies + num_columns_per_row - 1) // num_columns_per_row

    for row in range(num_rows):
        cols = st.columns(num_columns_per_row)
        for i in range(num_columns_per_row):
            movie_index = row * num_columns_per_row + i
            if movie_index < num_movies:
                movie = movies.iloc[movie_index]
                with cols[i]:
                    if 'poster_path' in movie and pd.notna(movie['poster_path']):
                        image_url = 'https://image.tmdb.org/t/p/original' + movie['poster_path']
                    else:
                        image_url = 'https://www.phipix.com/data_rojet2/affiche-film-sans-visuel.jpg'  # URL d'une image de remplacement
                    
                    # Affichage des informations du film
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <img src="{image_url}" width="100" style="border-radius: 8px;">
                        <div style="text-align: center; line-height: 1.2; margin-top: 5px;">
                            <strong>{movie['title']}</strong><br></div>
                            <div style="text-align: center; line-height: 1.2; margin-bottom: 10px; font-size: 10px;">
                            Année : {movie['year']}<br>
                            """, unsafe_allow_html=True)

                    # Utiliser 'tconst' comme identifiant unique pour chaque bouton
                    button_key = f"details_button_{movie['tconst']}"
                    if st.button("Détails", key=button_key):
                        button_dialog(movie['tconst'], image_url)
                        
# Afficher les détails du film dans une boîte de dialogue
@st.dialog("Cast your vote")
def button_dialog(item, image_url):
    movie = df_ml_reco.loc[df_ml_reco['tconst']==item]
    st.dialog(f"Détails pour {movie['title']}"):
    st.image(image_url, width=300)
    st.write(f"**Titre :** {movie['title']}")
    st.write(f"**Année :** {movie['year']}")
    st.write(f"**Runtime :** {movie.get('runtime', 'N/A')}")
    st.write(f"**Average Rating :** {movie.get('averageRating', 'N/A')}")
    st.write(f"**Number of Votes :** {movie.get('numVotes', 'N/A')}")
    st.write(f"**Description :** {movie.get('description', 'N/A')}")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.phipix.com/data_projet2/banniere-reco-cine-creuse.png");

background-position: center top 60px;
background-repeat: no-repeat;
background-color: #262730; /* couleur de fond */
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(38,39,48,1);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Interface Streamlit

# Titre de l'application // Ciné Creuse - Recommandation de films
st.title(' ')
st.title(' ')
st.title(' ')

# Container fixe pour la zone de saisie de texte
search_container = st.container()
result_container = st.container()
selection_container = st.container()
footer_container = st.container()

with search_container:

    # Création de colonnes pour aligner le selectbox et le bouton radio côte à côte
    col1, col2 = st.columns([8, 3])

    with col1:
        # Entrée utilisateur avec suggestions automatiques, initialisée vide
        options = [""] + df_ml_reco['title'].tolist()
        selected_title = st.selectbox('Recherchez un film que vous aimez : 👇', options)

    with col2:
        result_nb20 = st.radio(
            "Nbr de suggestions : 👇",
            [10, 15, 20],
            index=0,
            key="chk_result_nb20",
            horizontal=True
        )

if selected_title:
    with result_container:
        similar_movies = find_similar_movies(selected_title, knn, df_ml_reco, final_features, n_neighbors=result_nb20 + 1)
        if similar_movies is not None:
            st.write(f'Films similaires à: **{selected_title}**: 👇')

            # Exclure le film sélectionné des résultats
            similar_movies = similar_movies[similar_movies['title'] != selected_title]

            # Afficher les films similaires
            display_movies(similar_movies)

with selection_container:
    st.markdown("## Notre sélection")
    tabs = st.tabs(["Policier", "Mystère", "Familiale", "Historique", "Biographique", "Drame", "Western", "Guerre", "Action", "Comédie"])

    with tabs[0]:
        st.header("Policier")
        crime_movies = df_ml_reco[df_ml_reco['Crime'] == 1].sample(n=10)
        display_movies(crime_movies)
        
    with tabs[1]:
        st.header("Mystère")
        mystery_movies = df_ml_reco[df_ml_reco['Mystery'] == 1].sample(n=10)
        display_movies(mystery_movies) 

    with tabs[2]:
        st.header("Familiale")
        family_movies = df_ml_reco[df_ml_reco['Family'] == 1].sample(n=10)
        display_movies(family_movies)

    with tabs[3]:
        st.header("Historique")
        history_movies = df_ml_reco[df_ml_reco['History'] == 1].sample(n=10)
        display_movies(history_movies)

    with tabs[4]:
        st.header("Biographique")
        biography_movies = df_ml_reco[df_ml_reco['Biography'] == 1].sample(n=10)
        display_movies(biography_movies)

    with tabs[5]:
        st.header("Drame")
        drama_movies = df_ml_reco[df_ml_reco['Drama'] == 1].sample(n=10)
        display_movies(drama_movies) 

    with tabs[6]:
        st.header("Western")
        western_movies = df_ml_reco[df_ml_reco['Western'] == 1].sample(n=10)
        display_movies(western_movies)

    with tabs[7]:
        st.header("Guerre")
        war_movies = df_ml_reco[df_ml_reco['War'] == 1].sample(n=10)
        display_movies(war_movies)

    with tabs[8]:
        st.header("Action")
        action_movies = df_ml_reco[df_ml_reco['Action'] == 1].sample(n=10)
        display_movies(action_movies)

    with tabs[9]:
        st.header("Comédie")
        comedy_movies = df_ml_reco[df_ml_reco['Comedy'] == 1].sample(n=10)
        display_movies(comedy_movies)
        
    # Initialiser st.session_state.value si elle n'existe pas
    container = st.container()
    if 'Recharger une nouvelle sélection' not in st.session_state:
        st.session_state.value = " "
        if st.button("Recharger une nouvelle sélection"):
            st.session_state.value = " "
        container.header(st.session_state.value)

with footer_container:
    st.markdown("""
    <div style="text-align: center;">
        <img src='https://www.phipix.com/data_projet2/Logo-data-competence-200px.png' width='150' style='display: block; margin: 10px auto;'>
        <p>Dathanos™ 2024</p>
    </div>
""", unsafe_allow_html=True)

