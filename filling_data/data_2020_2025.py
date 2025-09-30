import requests
import csv
import time
import threading
import os
from typing import List, Dict

#Palabras que queremos incluir
KEYWORDS = [
    "historical fiction", "mystery", "thriller", "romance", "fantasy",
    "science fiction", "horror", "young adult", "adventure",
    "contemporary fiction", "literary fiction", "graphic novels",
    "comics", "manga", "high fantasy", "urban fantasy", "romantasy",
    "steampunk", "magical realism", "cyberpunk", "space opera",
    "dystopian", "crime", "detective", "spy", "espionage", "western",
    "war", "gothic", "victorian gothic", "memoir", "autobiography",
    "true crime", "poetry", "lgbt", "queer", "gay", "lesbian",
    "childrens", "fairy tales", "folklore", "mythology", "dark fantasy",
    "sword and sorcery", "supernatural", "occult", "vampires", "werewolves",
    "zombies", "ghost stories", "mythic fiction", "speculative fiction",
    "alternate history", "historical mystery", "historical romance",
    "historical fantasy", "romantic comedy", "chick lit", "new adult",
    "campus romance", "sports romance", "paranormal romance", "dark romance",
    "erotic romance", "romantic suspense", "historical adventure",
    "fae fantasy", "fey romance", "fairy tale retellings",
    "magical realism romance", "time travel romance", "space fantasy",
    "gothic romance", "reverse harem", "booktok", "fan fiction",
    "fandom culture", "feminist fiction", "dragons", "elves", "witches",
    "wizards", "pirates", "space fantasy", "villain origin stories",
    "antiheroes", "heist", "survival", "quests", "epic fantasy",
    "cozy mystery", "noir", "crime thriller", "legal thriller",
    "psychological thriller", "medical thriller", "disaster fiction",
    "apocalyptic", "post-apocalyptic", "AI fiction", "robot fiction",
    "time travel", "dimension travel", "portal fantasy", "virtual reality",
    "gamelit", "litRPG", "bromance", "found family", "enemies to lovers",
    "grumpy sunshine", "small town romance", "slow burn romance",
    "forbidden romance", "holiday romance", "summer romance", 
    "friends to lovers", "second chance romance", "fake relationship",
    "opposites attract", "age gap romance", "royal romance",
    "historical romance", "historical fantasy romance", "historical fiction",
    "historical mystery", "historical thriller", "historical adventure",
    "historical drama", "historical horror", "historical science fiction",
    "fallen angels", "fallen angels romance", "fallen angels fantasy",
    "fallen angels horror", "fallen angels thriller", "fallen angels mystery", 
    "fallen angels drama", "fallen angels science fiction",
    "fallen angels comedy", "fallen angels fantasy romance",
    "fallen angels adventure", "fallen angels horror romance",
    "fallen angels thriller romance", "fallen angels mystery romance",
    "fallen angels drama romance", "fallen angels science fiction romance",
    "fallen angels comedy romance", "fallen angels fantasy adventure",
    "fallen angels horror adventure", "fallen angels thriller adventure",
    "fallen angels mystery adventure", "fallen angels drama adventure",
    "fallen angels science fiction adventure", "fallen angels comedy adventure",
    "historical fantasy adventure", "historical fantasy romance",
    "fae fantasy romance", "fairy tale fantasy", "magical realism fantasy",
    "time travel fantasy", "space fantasy romance", "gothic fantasy",
    "gothic romance", "reverse harem fantasy", "booktok fantasy",
    "fan fiction fantasy", "fandom culture fantasy", "feminist fantasy",
    "dragons fantasy", "elves fantasy", "witches fantasy", 
    "witches romance", "antiheroes fantasy", "heist fantasy",
    "survival fantasy", "quests fantasy", "epic fantasy romance",
    "cozy mystery fantasy", "noir fantasy", "crime thriller fantasy",
    "legal thriller fantasy", "psychological thriller fantasy",
    "sirens fantasy", "mermaids fantasy", "vampires fantasy",
    "distopian fantasy", "post-apocalyptic fantasy",
    "AI fantasy", "robot fantasy", "time travel fantasy",
    "dimension travel fantasy", "portal fantasy romance",
    "zombies", "ghosts fantasy", "supernatural fantasy",
    "occult fantasy", "occult romance", "occult thriller",
    "occult horror", "occult mystery", "occult adventure",
    "occult science fiction", "occult drama", "occult comedy",
    "occult drama", "occult science fiction", "occult adventure",
    "occult comedy", "occult drama", "occult science fiction",
    "romance thriller", "romance mystery", "romance drama",
    "romance horror", "romance adventure", "romance science fiction",
    "romance comedy", "romance fantasy", "romance drama",
    "occult fantasy", "vampires romance", "vampires fantasy",
    "vampires horror", "werewolves romance", "werewolves horror",
    "werewolves fantasy", "zombies romance", "zombies horror",
    "zombies fantasy", "ghost stories fantasy","literary fiction fantasy",
    "literary fiction romance", "literary fiction horror",
    "dragons fantasy", "dragons romance", "dragons horror",
    "mythic fiction fantasy", "speculative fiction fantasy",
    "alternate history fantasy", "historical mystery fantasy",
    "yaoi fantasy", "yaoi romance", "yuri fantasy",
    "yuri romance", "shoujo fantasy", "shoujo romance",
    "shounen fantasy", "shounen romance", 
    "morally grey characters fantasy", "morally grey characters romance",
    "elves romance", "fairies fantasy", "fairies romance",
    "wizards fantasy", "pirates fantasy", "villain origin stories fantasy",
    "historical fantasy mystery", "historical fantasy thriller",
    "historical fantasy drama", "historical fantasy horror",
    "historical fantasy science fiction", "historical fantasy adventure romance",
    "historical fantasy adventure mystery", "historical fantasy adventure thriller",
    "danmei", "yaoi", "yuri", "shoujo", "shounen",
    "isekai fantasy", "isekai adventure", "isekai romance", 
    "isekai harem", "isekai reverse harem", "isekai comedy",
    "isekai drama", "isekai action", "isekai fantasy romance",
    "isekai slice of life", "isekai mystery", "isekai thriller",
    "isekai horror", "isekai supernatural", "isekai sci-fi",
    "isekai dystopian", "isekai post-apocalyptic", "isekai historical",
    "isekai magical realism", "isekai steampunk", "isekai cyberpunk",
    "isekai space opera", "isekai dark fantasy", "isekai high fantasy",
    "isekai urban fantasy", "isekai romantic comedy", "isekai fantasy adventure",
    "isekai fantasy comedy", "isekai fantasy drama", "isekai fantasy action",
    "YA fantasy romance", "monster romance", "LGBT romance",
    "dragonslayer fiction", "slice of life", "boarding school fiction", "romcom",
    "feel good fiction", "superhero fiction", "urban legends",
    "haunted house fiction", "detective girls", "magical boarding schools",
    "witch academy", "academy fantasy", "academy romance", "time loop fiction",
    "clones", "mutants", "superpowers", "chosen one", "cursed romance",
    "morally grey characters", "beast romance", "villain romance",
    "villainess reborn", "isekai romance", "isekai fantasy", "isekai adventure",
    "isekai harem", "isekai reverse harem", "medieval fantasy",
    "court politics fantasy", "magic school", "monster hunter",
    "magical creatures", "animal companions", "alchemy fantasy",
    "dreamwalker fiction", "parallel worlds", "sci-fi romance",
    "cosmic horror", "dark academia", "fey fantasy", "fae romance",
    "retellings", "fairy tale retellings", "bookish romance",
    "introvert romance", "emotionally damaged romance", "healing romance",
    "revenge fantasy", "forbidden powers", "magical tournament",
    "survival games", "romantic suspense", "royalty romance",
    "reincarnation fiction", "Wattpad", "Wattpad books", "Wattpad story"
]

# Palabras clave que queremos excluir
BLACKLIST = {
    "medicine", "health", "psychology", "self_help", "religion", "spirituality",
    "philosophy", "politics", "history", "biography", "business", "economics",
    "finance", "law", "education", "technology", "programming", "travel",
    "engineering", "mathematics", "physics", "chemistry", "gardening",
    "biology", "environment", "nature", "travel", "food", "cooking",
    "gardening", "crafts", "hobbies", "sports", "fitness", "hobbies"
    "healthcare", "parenting", "pets", "animals", "home_improvement",
    "interior_design", "fashion", "beauty", "lifestyle", "cookingbooks",
    "personal_development", "relationships", "dating", "marriage",
    "education", "family", "spiritual_growth", "motivation", "inspiration",
    "games & activities", "art & photography", "music & performing arts",
    "theater & drama", "film & video", "television & radio", "essays & anthologies",
    "science", "success", "art", "music", "theater", "film", "television",
    "radio", "essays", "anthologies", "plays", "self_help",
    "spirituality", "philosophy", "politics", "history", "biography",
    "business", "economics", "finance", "law", "education", 
    "technology", "computers", "engineering", "mathematics", "physics",
    "chemistry", "biology", "environment", "nature", "travel", "food",
    "cooking", "gardening", "crafts", "hobbies", "sports", "fitness",
    "healthcare", "parenting", "pets", "animals", "home_improvement",
    "interior_design", "fashion", "beauty", "lifestyle", "cookingbooks",
    "personal_development", "relationships", "dating", "marriage", "education",
    "family", "spiritual_growth", "motivation", "inspiration", "games & activities",
    "art & photography", "music & performing arts", "theater & drama", 
    "film & video", "television & radio", "essays & anthologies", "science" 
    "success", "art", "music", "theater", "film", "television", "radio",
    "essays", "anthologies","plays", "tourism", "travel guides", "cultural studies",
    "geography", "world history", "archaeology", "anthropology",
}
# Archivo CSV de salida
OUTPUT_FILE = "books_2020_to_2025.csv"
TARGET_TOTAL = 25000
MAX_THREADS = 30

COLUMNS = [
    "Title", "Author(s)", "ISBN", "Published", "Pages", "Publisher",
    "Language", "Description", "Categories", "Average Rating",
    "Ratings Count", "Thumbnail", "Info Link", "Google ID", "Genre",
    "Fetched From", "Format", "Series", "Physical Format", "Release Date"
]

books_lock = threading.Lock()
all_books: List[Dict] = []

# Reanudar si existe
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_books.append(row)
    print(f"Reanudando desde {len(all_books)} libros ya guardados.")
else:
    print("Iniciando nueva colección de libros por keyword...")

existing_isbns = {book["ISBN"] for book in all_books if book.get("ISBN")}

# Filtro de keywords usando la blacklist
filtered_keywords = [kw for kw in KEYWORDS if not any(bl in kw.lower() for bl in BLACKLIST)]
print(f"Palabras clave filtradas: {len(filtered_keywords)}")

if not filtered_keywords:
    print("No quedan keywords tras aplicar la blacklist. Revisa las listas.")
    exit()


def fetch_books_by_keyword(keyword: str) -> List[Dict]:
    books: List[Dict] = []
    for lang in ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko']:
        for start_index in range(0, 800, 40):
            params = {
                "q": keyword,
                "startIndex": start_index,
                "maxResults": 40,
                "printType": "books",
                "langRestrict": lang
            }
            try:
                resp = requests.get(
                    "https://www.googleapis.com/books/v1/volumes",
                    params=params,
                    timeout=10
                )
                if resp.status_code != 200:
                    continue

                items = resp.json().get("items", [])
                for item in items:
                    volume = item.get("volumeInfo", {})

                    #Filtrar por año 2020-2025
                    pub_date = volume.get("publishedDate", "")
                    if not any(pub_date.startswith(str(y)) for y in range(2020, 2026)):
                        continue

                    cats = [
                        c.lower().replace(" ", "_")
                        for c in volume.get("categories", [])
                    ]
                    if any(cat in BLACKLIST for cat in cats):
                        continue

                    identifiers = volume.get("industryIdentifiers", [])
                    isbn = next(
                        (i["identifier"] for i in identifiers if "ISBN" in i.get("type", "")),
                        None
                    )
                    if not isbn or isbn in existing_isbns:
                        continue

                    book = {
                        "Title": volume.get("title"),
                        "Author(s)": ", ".join(volume.get("authors", [])),
                        "ISBN": isbn,
                        "Published": pub_date,
                        "Pages": volume.get("pageCount"),
                        "Publisher": volume.get("publisher"),
                        "Language": volume.get("language"),
                        "Description": volume.get("description"),
                        "Categories": ", ".join(volume.get("categories", [])),
                        "Average Rating": volume.get("averageRating"),
                        "Ratings Count": volume.get("ratingsCount"),
                        "Thumbnail": volume.get("imageLinks", {}).get("thumbnail"),
                        "Info Link": volume.get("infoLink"),
                        "Google ID": item.get("id"),
                        "Genre": keyword,
                        "Fetched From": "Google Books",
                        "Format": None,
                        "Series": None,
                        "Physical Format": None,
                        "Release Date": pub_date
                    }
                    books.append(book)
                    existing_isbns.add(isbn)

            except Exception:
                continue
            time.sleep(0.1)
    return books


def worker(keyword: str):
    found = fetch_books_by_keyword(keyword)
    with books_lock:
        all_books.extend(found)
        print(f"{len(found)} libros encontrados en '{keyword}'")


def save_csv():
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_books)


#Bucle principal hasta alcanzar el total
while len(all_books) < TARGET_TOTAL:
    for i in range(0, len(filtered_keywords), MAX_THREADS):
        batch = filtered_keywords[i: i + MAX_THREADS]
        threads = []
        for kw in batch:
            t = threading.Thread(target=worker, args=(kw,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        save_csv()
        print(f"Guardados {len(all_books)} libros en {OUTPUT_FILE}")

        if len(all_books) >= TARGET_TOTAL:
            print("¡Has alcanzado los 25.000 libros! Puedes parar si quieres.")
            break

    print("Reintentando keywords para seguir buscando...")
    time.sleep(10)

print("Proceso finalizado. Total de libros:", len(all_books))
