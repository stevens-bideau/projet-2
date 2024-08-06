import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Intégrer le CSS pour l'image de fond
background_image_url = "https://www.phipix.com/data_projet2/banniere-reco-cine-creuse.png"  # Remplacez par l'URL de votre image
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("{background_image_url}");
    background-position: center top 60px; /* caller l'image en centré en haut (top) */
    background-repeat: no-repeat; /* ne pas répéter l'image en mosaïac */
    background-color: #262730; /* couleur de fons */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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

# Fonction pour afficher des films avec un style
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
                    image_url = 'https://image.tmdb.org/t/p/original' + movie['poster_path']
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <img src="{image_url}" width="100" style="border-radius: 8px;">
                        <div style="text-align: center; line-height: 1.2; margin-top: 5px;">
                            <strong>{movie['title']}</strong><br></div>
                            <div style="text-align: center; line-height: 1.2; margin-bottom: 10px; font-size: 10px;">
                            Année : {movie['year']}<br>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Interface Streamlit

# Titre de l'application // Ciné Creuse - Recommandation de films
st.title(' ')
st.title(' ')
st.title(' ')

# Container fixe pour la zone de saisie de texte
search_container = st.container()
result_container = st.container()
selection_container = st.container()

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
    tabs = st.tabs(["Policier", "Historique", "Drame", "Action", "Comédie"])

    with tabs[0]:
        st.header("Policier")
        crime_movies = df_ml_reco[df_ml_reco['Crime'] == 1].sample(n=10, random_state=42)
        display_movies(crime_movies)

    with tabs[1]:
        st.header("Historique")
        crime_movies = df_ml_reco[df_ml_reco['Hystory'] == 1].sample(n=10, random_state=42)
        display_movies(crime_movies)

    with tabs[2]:
        st.header("Drame")
        crime_movies = df_ml_reco[df_ml_reco['Drama'] == 1].sample(n=10, random_state=42)
        display_movies(crime_movies)

    with tabs[3]:
        st.header("Action")
        crime_movies = df_ml_reco[df_ml_reco['Action'] == 1].sample(n=10, random_state=42)
        display_movies(crime_movies)

    with tabs[4]:
        st.header("Comédie")
        crime_movies = df_ml_reco[df_ml_reco['Comedy'] == 1].sample(n=10, random_state=42)
        display_movies(crime_movies)

