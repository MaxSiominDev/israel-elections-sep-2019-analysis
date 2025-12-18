from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

ARABIC_HINTS = [
    "umm", "abu", "ibn", "al-", "al ", "el-", "el ",
    "deir", "dayr", "sheikh", "shaykh", "kafr", "arraba", "ain", "ein", "tribe",
]

def rule_is_arabic(city: str) -> bool:
    c = city.lower()
    return any(p in c for p in ARABIC_HINTS)

cities = [
    "Umm al-Fahm", "Kafr Qasim", "Deir al-Asad", "Arraba", "Sakhnin",
    "Tamra", "Baqa al-Gharbiyye", "Baqa al-Sharqiyya", "Nazareth Illit",
    "Shfaram", "Tira", "Fureidis", "Kafr Manda", "Kafr Yasif",
    "I'billin", "Daliyat al-Karmel", "Jish", "Majd al-Krum", "Sajur",
    "Bir al-Maksur", "Kfar Hananya", "Tur'an", "Yanuh-Jat", "Kfar Kama",
    "Abu Snan", "Jatt", "Basmat Tab'un", "Eilabun", "Shefa-'Amr",
    "Taibe", "Kfar Qara", "Mas'ade", "Jaljulia", "Umm al-Kheir",
    "Deir Hanna", "Ein al-Asad", "Kafr Kanna", "Rameh", "Nahf",
    "Isfiya", "Jisr az-Zarqa", "Kafr Bara", "Kafr Sur", "Ba'ana",
    "Barta'a", "Kafr Yasif", "Al-Zarqa", "Kafr Qasim", "Kafr Manda",
    "Iksal", "Fassuta", "Bir al-Maksur", "Tur'an", "Rumat al-Heib",
    "Khirbat al-Mansura", "Al-Birwa", "Al-Jalama", "Al-Mansura",
    "Al-Maghar", "Kafr Kama", "Yanuh-Jat", "Arrabe", "Baqa al-Gharbiyye",
    "Baqa al-Sharqiyya", "Kafr Qara", "Deir al-Asad", "Umm al-Fahm",
    "Sakhnin", "Deir Hanna", "I'billin", "Jish", "Shefa-'Amr",
    "Majd al-Krum", "Rameh", "Kafr Kanna", "Nahf", "Tur'an",
    "Kafr Yasif", "Ein al-Asad", "Kafr Manda", "Jatt", "Basmat Tab'un",
    "Eilabun", "Kafr Bara", "Taibe",
]

labels = [1] * len(cities)

cities += [
    "Tel Aviv", "Jerusalem", "Haifa", "Ramat Gan", "Givatayim",
    "Petah Tikva", "Rishon LeZion", "Rehovot", "Ashdod", "Beersheba",
    "Kfar Saba", "Netanya", "Eilat", "Modiin", "Holon", "Bat Yam",
    "Raanana", "Hod Hasharon", "Lod", "Ramla", "Yavne", "Or Akiva",
    "Hadera", "Akko", "Kiryat Shmona", "Maale Adumim", "Afula",
    "Nahariya", "Kiryat Gat", "Kiryat Malakhi", "Dimona", "Givat Shmuel",
    "Kiryat Ono", "Even Yehuda", "Mazkeret Batya", "Sderot", "Yehud",
    "Kiryat Motzkin", "Rosh HaAyin", "Karmiel", "Arad", "Nahshonim",
    "Afula", "Ofakim", "Beit Shemesh", "Givat Ada", "Zikhron Yaakov",
    "Tsfat", "Kiryat Bialik", "Ramat HaSharon", "Kiryat Yam", "Kiryat Haim",
    "Kiryat Tivon", "Or Yehuda", "Migdal HaEmek", "Shlomi", "Yehud-Monoson",
    "Sakhnin", "Dimona", "Ramat HaNegev", "Kiryat Malakhi", "Nesher",
    "Yokneam", "Sdot Yam", "Kiryat Ekron", "Omer", "Mazra'a", "Kadima",
]

labels += [0]*(len(cities)-len(labels))

assert len(cities) == len(labels), f"{len(cities)} != {len(labels)}"

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(cities)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

def is_arabic_settlement_name(city_name: str) -> bool:
    if rule_is_arabic(city_name):
        return True
    x = vectorizer.transform([city_name])
    return clf.predict(x)[0] == 1
