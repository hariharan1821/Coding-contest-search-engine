import ast
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, render_template, request

app = Flask(__name__, template_folder= 'template')

with open('data_processing/documents.txt', 'r', encoding='utf-8') as f:
    for i in f:
        documents = ast.literal_eval(i)
        break

with open('data_processing/links.txt', 'r', encoding='utf-8') as f:
    for i in f:
        links = ast.literal_eval(i)
        break


for i in range(len(documents)):
        documents[i] = " ".join(documents[i])
    
vectorizer = TfidfVectorizer() 
vectors = vectorizer.fit_transform(documents)

tf_idf = pd.DataFrame(vectors.todense()).iloc[:-1]  
tf_idf.columns = vectorizer.get_feature_names_out()
tfidf_matrix = tf_idf.T
tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)
tfidf_matrix = tfidf_matrix.sort_values(by ='count')

names = list(tfidf_matrix['count'].index.values)

def getresults(query):

    finallinks = []
    st = LancasterStemmer()
    c_query = [st.stem(i.lower()) for i in query.split(" ")]
    cs = [0]*len(names) 

    for i in range(len(names)):
        for j in range(len(c_query)):
            if names[i] == c_query[j]:
                cs[i]+=1

    cs_df = pd.DataFrame(cs)

    rank = []
    for i in range(tfidf_matrix.shape[1]-1):
        rank.append((i, cosine_similarity(cs_df.T, pd.DataFrame(tfidf_matrix[i]).T)[0]))

    
    ind = sorted(rank, key = lambda x:x[1], reverse = True)[:20]
    if ind[0][1]==0:
        # print("Not found")
        finallinks = ["Not found"]
        pass
    else:
        for i,j in ind:
            if j!=0:
                finallinks.append(links[i][0])

    return finallinks



@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method=="GET":
        result = []
        return render_template("index.html", result = result, len = len(result))
    else:
        query = request.form.get('search')
        result = getresults(query)
        return render_template("index.html", result = result, len = len(result))

if __name__=='__main__':
    app.run()



