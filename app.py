from flask import Flask, request, jsonify
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from flask_cors import CORS




app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines (dans un environnement de développement, pour la production, spécifiez les origines autorisées)

# Charger les stop words en français
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))


# Prétraitement des données
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language='french')
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Charger les données d'offres d'emploi à partir d'un fichier CSV
df = pd.read_csv("offres.csv", delimiter=";")

# Prétraitement des colonnes
df['competences_preprocessed'] = df['Competences_requises'].apply(preprocess_text)
df['description_preprocessed'] = df['Description'].apply(preprocess_text)
df['niveau_etude_preprocessed'] = df['Niveau_detudes_requis'].apply(preprocess_text)
df['experience_preprocessed'] = df['Experience_requise'].apply(preprocess_text)

# Concaténer les colonnes pertinentes en une seule chaîne de texte
df['combined_text'] = df['competences_preprocessed'] + ' ' + df['description_preprocessed'] + ' ' + df[
    'niveau_etude_preprocessed'] + ' ' + df['experience_preprocessed']

# Créer une instance de CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_text'])

# Créer une instance du modèle de clustering K-means
kmeans_model = KMeans(n_clusters=3)

# Entraîner le modèle de clustering sur les vecteurs de caractéristiques
kmeans_model.fit(X.toarray())

# Attribuer des clusters aux offres d'emploi
job_clusters = kmeans_model.predict(X.toarray())

# Ajouter les informations des clusters aux données d'offres d'emploi
df['cluster'] = job_clusters


@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    data = request.json

    # Prétraitement des informations du profil du candidat
    profil_candidat = data['competences'] + ' ' + data['description'] + ' ' + data['niveau_etudes'] + ' ' + data[
        'experience']
    profil_candidat_preprocessed = preprocess_text(profil_candidat)

    # Calcul de la similarité cosinus avec le profil du candidat
    profil_candidat_features = vectorizer.transform([profil_candidat_preprocessed])
    similarities = cosine_similarity(profil_candidat_features, X.toarray()).flatten()

    # Ajout des scores de similarité au DataFrame
    df['similarity'] = similarities * 100

    # Trier les offres par similarité décroissante
    df_sorted = df.sort_values(by='similarity', ascending=False)

    # Sélectionner les meilleures offres
    meilleures_offres = df_sorted.head(10)

    # Convertir les résultats en format JSON avec des informations supplémentaires
    result = meilleures_offres[
        ["Titre", "similarity", "cluster", "Competences_requises", "Niveau_detudes_requis",
         "Experience_requise", "Salaire", "Type_de_contrat"]].to_dict(orient='records')

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
