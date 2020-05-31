import pandas as pd

category_tags = [
    #'Art',
    #'Biography'
    #'Business',
    #"Chick-Lit",
    "Children-s",
    #'Christian',
    'Classics',
    'Comics',
    'Contemporary',
    'Cookbooks',
    'Crime',
    'Fantasy',
    'Fiction',
    #'Graphic-Novels',
    #'Historical-Fiction',
    'History',
    'Horror',
    'Humor-and-Comedy',
    'Manga',
    'Memoir',
    'Music',
    'Mystery',
    'Nonfiction',
    'Paranormal',
    'Philosophy',
    'Poetry',
    'Psychology',
    #'Religion',
    'Romance',
    'Science',
    'Science-Fiction',
    #'Self-Help',
    #'Suspense'
    'Spirituality',
    'Sports',
    'Thriller',
    'Travel',
    #'Young-Adult'
    ]

def get_categories():

    tags = pd.read_csv('./goodreads10k/tags.csv')
    tags_dict = {}
    for idx, row in tags.iterrows():
        tags_dict[row['tag_id']] = row['tag_name']

    book_tags = pd.read_csv('./goodreads10k/book_tags.csv')
    book_categories = {}

    categories = [t.lower() for t in category_tags]
    c_tags = tags[tags.tag_name.isin(categories)]

    category_book_tags = book_tags[ book_tags.tag_id.isin(c_tags.tag_id)]
    most_popular_tags_idx = (category_book_tags.groupby('goodreads_book_id')['count'].transform(max) == category_book_tags['count'])

    for idx, row in category_book_tags[most_popular_tags_idx].iterrows():
        book_categories[row['goodreads_book_id']] = tags_dict[row['tag_id']]

    return book_categories

if __name__ == '__main__':
    print(get_categories())