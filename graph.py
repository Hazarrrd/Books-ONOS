import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


book_columns = ['isbn', 'id', 'name', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
book_columns[1] = 'id'
book_columns[2] = 'name'
data1 = pd.read_csv('books1000_10000.txt',error_bad_lines=False, sep=", ", names=book_columns)
data2 = pd.read_csv('books10000-11000.txt',error_bad_lines=False, sep=", ", names=book_columns)
books = pd.concat([data1, data2], ignore_index=True)
books_id_to_name = {row['id'] : row['name'] for ix, row in books[['id', 'name']].iterrows()}
#print(2048 in books_id_to_name)

tags = pd.read_csv('tags1000-10000.txt',error_bad_lines=False, sep=", ", names=['book_id', 'tag'])
tags2 = pd.read_csv('10000-11000.txt',error_bad_lines=False, sep=", ", names=['book_id', 'tag'])
book_tags = pd.concat([tags, tags2], ignore_index=True)


G = nx.Graph()

for tag, tag_idx in book_tags.groupby(['tag']).groups.items():
    if len(tag_idx) < 300:
        continue

    print(tag)
    book_idxs = book_tags.iloc[tag_idx, :].book_id.values
    #print('books idx' ,book_idxs)
    for i in range(len(book_idxs)):
        for j in range(i+1, len(book_idxs)):
            
            idx1 = book_idxs[i]
            idx2 = book_idxs[j]

            try:
                book1 = books_id_to_name[idx1]
                book2 = books_id_to_name[idx2]
            except:
                continue

            if not G.has_edge(idx1, idx2):
                G.add_edge(book1, book2, weight=1)
            else:
                G[idx1][idx2]['weight'] = G[idx1][idx2]['weight'] + 1
    

nx.draw(G, with_labels=False)
plt.show()