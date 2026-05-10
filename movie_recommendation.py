import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
# Merge datasets
movies = movies.merge(credits, on="title")

# Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew','vote_average']]

# Helper functions
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Data preprocessing
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags','vote_average']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)


# Movie recommendation
def recommend_by_movie(movie, num):

    movie = movie.lower()

    if movie not in new_df['title'].str.lower().values:
        print("Movie not found")
        return

    movie_index = new_df[new_df['title'].str.lower()==movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])

    start = 1

    while start < len(movie_list):

        end = start + num

        print(f"\nRecommendations {start} to {end-1}\n")

        for i,m in enumerate(movie_list[start:end]):
            index = m[0]
            title = new_df.iloc[index].title
            rating = new_df.iloc[index].vote_average

            print(f"{start+i}. {title} ⭐ {rating}")

        start = end

        choice = input("\nDo you want more recommendations? (yes/no): ").lower()

        if choice == "no":
            break


# Genre recommendation
def recommend_by_genre(genre, num):

    genre = genre.lower()

    genre_movies = movies[movies['genres'].astype(str).str.lower().str.contains(genre)]

    genre_movies = genre_movies.sort_values(by='vote_average', ascending=False)

    start = 0

    while start < len(genre_movies):

        end = start + num

        print(f"\nTop {genre.capitalize()} Movies {start+1} to {end}\n")

        subset = genre_movies.iloc[start:end]

        for i,row in enumerate(subset.itertuples()):
            print(f"{start+i+1}. {row.title} ⭐ {row.vote_average}")

        start = end

        choice = input("\nDo you want more recommendations? (yes/no): ").lower()

        if choice == "no":
            break


# Main Program
while True:

    print("\n🎬 Movie Recommendation System\n")

    print("1. Recommend by Movie")
    print("2. Recommend by Genre")
    print("3. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "3":
        print("Thank you for using the system!")
        break

    num = int(input("How many recommendations do you want? "))

    if choice == "1":

        movie = input("Enter your favourite movie: ")
        recommend_by_movie(movie, num)

    elif choice == "2":

        genre = input("Enter genre (Action, Comedy, Drama etc.): ")
        recommend_by_genre(genre, num)

    else:
        print("Invalid choice")