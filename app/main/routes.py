from flask import render_template, request
from . import main
import numpy as np

# Quick Pick imports
from app.models.quick_model import load_df, apply_multi_filter
from app.utils.sorting import sort_df_by_title_lns, sort_lns_iterable

# Smart Match imports  
from app.models.smart_model import load_df_knn, prepare_knn, knn_recommend

# Deep Dive imports
from app.models.deep_model import load_dataset, load_embedder, load_faiss_artifacts, semantic_recommend

# Global data loading
DATA_PATH = "data/final/books_translated.csv"

# Load data once at startup
df_quick, GENRES, TITLES, AUTHORS = load_df(DATA_PATH)
df_smart = load_df_knn(DATA_PATH)
embedding_matrix = np.load("data/final/embeddings.npy")
knn_model = prepare_knn(df_smart, embedding_matrix)

df_deep = load_dataset()
embedder = load_embedder()
faiss_index, embeddings = load_faiss_artifacts()

@main.route("/")
def index():
    """Home page with application overview"""
    return render_template("index.html")

@main.route("/quick-pick", methods=["GET", "POST"])
def quick_pick():
    """Quick search with multiple filters and pagination"""
    if request.method == "POST":
        # Extract filter parameters
        title_kw = request.form.get("title_kw", "").strip()
        author_kw = request.form.get("author_kw", "").strip()
        genre_kw = request.form.getlist("genre_kw")
        lang_kw = request.form.get("lang_kw", "")
        pub_kw = ""

        # Apply filters
        results = apply_multi_filter(df_quick, title_kw, author_kw, genre_kw, lang_kw, pub_kw)

        # Apply sorting
        order_by = request.form.get("order_by", "rating (high to low)")
        if order_by == "rating (high to low)":
            results = results.sort_values(by="rating", ascending=False)
        else:
            results = sort_df_by_title_lns(results)

        # Pagination logic
        page_size = int(request.form.get("page_size", 20))
        page = int(request.form.get("page", 1))
        n_pages = max(1, int(np.ceil(len(results) / page_size)))
        start = (page - 1) * page_size
        end = start + page_size

        # Prepare data for template
        page_results = results.iloc[start:end].to_dict(orient="records")
        total = len(results)

        filters = {
            "title_kw": title_kw,
            "author_kw": author_kw,
            "genre_kw": genre_kw,
            "lang_kw": lang_kw,
            "order_by": order_by
        }

    else:
        # Default values for GET request
        page_results = []
        filters = {
            "title_kw": "",
            "author_kw": "",
            "genre_kw": [],
            "lang_kw": "",
            "order_by": "rating (high to low)"
        }
        page = 1
        page_size = 20
        n_pages = 1
        total = 0

    return render_template("quick_pick.html",
        genres=GENRES,
        languages=sort_lns_iterable(df_quick["language"].unique()),
        results=page_results,
        total=total,
        page=page,
        n_pages=n_pages,
        page_size=page_size,
        filters=filters
    )

@main.route("/smart-match", methods=["GET", "POST"])
def smart_match():
    """AI-powered book recommendations based on similarity"""
    book_query = request.form.get("book_query", "").strip().lower()
    selected_book = request.form.get("selected_book", "")
    exclude_series = request.form.get("exclude_series") == "on"
    exclude_author = request.form.get("exclude_author") == "on"
    top_knn = int(request.form.get("top_knn", 5))

    filtered_titles = []
    results = []
    message = ""
    
    # Filter titles based on user input
    if book_query:
        filtered_titles = df_smart[
            df_smart["normalized_title"].str.contains(book_query)
        ]["title"].tolist()

    # Generate recommendations if a book is selected
    if request.method == "POST" and selected_book:
        results_df, message = knn_recommend(
            selected_book, exclude_series, exclude_author, top_knn,
            df_smart, embedding_matrix, knn_model
        )
        results = results_df.to_dict(orient="records")

    return render_template("smart_match.html",
        book_query=book_query,
        filtered_titles=filtered_titles,
        selected_book=selected_book,
        exclude_series=exclude_series,
        exclude_author=exclude_author,
        top_knn=top_knn,
        results=results,
        message=message
    )

@main.route("/deep-dive", methods=["GET", "POST"])
def deep_dive():
    """Semantic search based on book descriptions"""
    query = request.form.get("query", "").strip()
    language = request.form.get("language", "")
    min_rating = float(request.form.get("min_rating", 0.0))
    top_n = int(request.form.get("top_n", 5))

    results = []
    message = ""
    
    if request.method == "POST" and query:
        results_df, message = semantic_recommend(
            query, df_deep, embedder, faiss_index, top_n, language, min_rating
        )
        if results_df is not None:
            results = results_df.to_dict(orient="records")

    return render_template("deep_dive.html",
        query=query,
        language=language,
        min_rating=min_rating,
        top_n=top_n,
        results=results,
        message=message,
        languages=sort_lns_iterable(df_deep["language"].unique())
    )