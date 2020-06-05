import pandas as pd


def preprocessing(file_name):
    columns = ["ISBN", "ID", "TITLE", "COUNTRY", "YEAR", "MONTH", "RATE", "PAGES", "POPULARITY", "SHELVE", "AUTHOR"]
    df = pd.read_csv(file_name, error_bad_lines=False, sep=", ", names=columns, warn_bad_lines=False, engine='python')
    df = df[df.ISBN != "None"]
    df = df[df.ID != "None"]

    df.drop('MONTH', axis=1, inplace=True)
    df.drop('SHELVE', axis=1, inplace=True)
    df.drop('ISBN', axis=1, inplace=True)
    df.drop('COUNTRY', axis=1, inplace=True)

    df = df.drop_duplicates(subset="ID", keep="first")
    df = df.drop_duplicates(subset="TITLE", keep="first")
    return df


def shelves_to_dict(file_name):
    dic = {}
    bad_tags = ['home-library', 'audiobook', 'currently-reading', 'favourites', 'own-it',
                'to-read', 'owned', 'literature', 'to-buy', 'i-own', 'nonfiction', 'books-i-own', 'my-library',
                'ebooks', 'default', 'wish-list', 'abandoned', 'library',
                'favorites',  'owned-books', 'ebook', 'kindle', 'my-books',
                'calibre', 'audio_wanted', 'on-hold', 'on-my-shelf', 'personal-library',  'did-not-finish','to-read-fiction',
                'tbr', 'unread', 'audio-books', 'books', 'maybe', 'to-read-non-fiction', 'audio', 'dnf', 'unfinished', 'must-read',
                'book-club', 'bookshelf', 'e-books', 're-read', 'audible', 'audiobooks', 'have',
                 'e-book', 'didn-t-finish', 'read-in-2014', 'read-in-2015', 'partially-read', '2006', 'favorite', 'shelved',
                'want-to-buy', 'general', 'owned-to-read', 'read-in-2017', 'bookclub', 'read-in-2018', 'books-i-have', 'finished',
                'read-in-english', 'hardcover', 'read-in-2016', 'borrowed', 'my-bookshelf', '1', 'reference', 'not-interested',
                'in-my-library', 'other', 'on-the-shelf', 'audio-book', 'own-to-read', '1001-books', 'in-translation', 'bought', 'not-read',
                'lit', '1001', '5-stars', 'suspense', 'reviewed', 'to-re-read', 'read-in-2019']
    with open(file_name, 'r') as f:
        for line in f:
            id, shelve = line.strip().split(', ')
            if shelve not in bad_tags:
                if id in dic.keys() and shelve not in dic[id]:
                    dic[id].append(shelve)
                else:
                    dic[id] = [shelve]
    return dic

def delete_rare_tags(dic, limit):
    all_tags = []
    for key, value in dic.items():
        all_tags += value
    dic_freq_tags = {}
    for i in all_tags:
        if i in dic_freq_tags.keys():
            dic_freq_tags[i] += 1
        else:
            dic_freq_tags[i] = 1

    list_uniqe_tags = []
    for key, value in dic_freq_tags.items():
        if value > limit:
            list_uniqe_tags.append(key)

    for key, value in dic.items():
        dic[key] = list(set(value) & set(list_uniqe_tags))
    return dic

def get_one():
    file_name = 'data/books1000_10000.txt'
    file_name_shelves = 'data/10000-11000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_second():
    file_name = "data/books10000-11000.txt"
    file_name_shelves = 'data/100000-102000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_third():
    file_name = "data/books100000-102000.txt"
    file_name_shelves = 'data/tags1000-10000.txt'
    return preprocessing(file_name), shelves_to_dict(file_name_shelves)


def get_all_data(limit=200):
    df1, shelv1 = get_one()
    df2, shelv2 = get_second()
    df3, shelv3 = get_third()

    frames = [df1, df2, df3]
    df = pd.concat(frames)
    df = df.drop_duplicates(subset="ID", keep="first")
    df = df.drop_duplicates(subset="TITLE", keep="first")
    shelv = {**shelv1, **shelv2, **shelv3}
    id_list = df["ID"].tolist()
    keys_to_delete = []
    for key in shelv.keys():
        if int(key) not in id_list:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del shelv[key]
    shelv = delete_rare_tags(shelv, limit=limit)

    return df, shelv