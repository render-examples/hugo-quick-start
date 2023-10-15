---
title: "My NLP Interview Experience (9 June 2023)"
date: 2023-06-09T10:15:27+08:00
tags: ['blog', 'nlp', 'machine learning']
---

A recent interview for a NLP position gave me an opportunity to delve into NLP problems and some good basic introductions to them. The topics that I looked into were related to semantic search and document deduplication. In this article (and with it, jupyter notebooks), I will dive into the domain semantic search.

For the interview, I focused on the topic of Semantic Search. Semantic Search consists of returning texts that are most closely related to a provided search query. These are not strictly speaking lexical based searches, where the most keywords matched shows up in the search results.

There are many ways to capture the semantic of a corpus, either from a pretrained model perspective or a more statistical approach, as well as how to scale it up to accomodate more data in a fast and reasonable manner. For the interview itself, I decided to go for TF-IDF based search as it was the easiest to explain within an hour, and also can be further explained for document deduplication. 

In the end, I did not pass the interview. Whilst I cannot speak on behalf of the interviewer as I was not given replies nor feedback about the interview, I think this is a good learning opportunity to explain my work and a future v2.0 of it. This post will go through the jupyter notebook

## **Data**

We see the data that we are working with here:

|  | jobId | jobUrl | jobTitle | jobDescription | datePosted | companyId | companyIdNormalised | companyName | rawWageMin | rawWageMax | sourceName | qualifications |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 00d9917e95ebbb58d237e90b5a01095947c31fbe119d82... | https://www.efinancialcareers.sg/jobs-Singapor... | Change Manager, IBOR Transition Programme | Change Manager, IBOR Transition Programme<br/>... | 2022-04-01 | 7cc4c2d8b4893e7c64265beccd30d4c1b644cf8b57a9e8... | 06634f73009b4765beafae5f98c0996b33870a3d34fa87... | Standard Chartered Bank | 0 | 0 | E-FinancialCareer | ['No Requirement'] |
| 1 | 0104963e8e1289488f2ff96edfe95dddc9ab84231b37a5... | https://www.efinancialcareers.sg/jobs-Singapor... | Analyst, KYC Analyst, Corporate Banking, Insti... | Analyst, KYC Analyst, Corporate Banking, Insti... | 2022-04-01 | c3475240458aa07566e1db7eec98affa5d85d8bd2b9577... | 1810faaf5f96a398f5b43df1b80809dbf5b7935f94a5f7... | DBS Bank Limited | 0 | 0 | E-FinancialCareer | ['Bachelors'] |
| 2 | 01561a39ff31372551e0be1caaf6a2c32150925f75af76... | https://www.jobstreet.com.sg/en/job/senior-leg... | (Senior) Legal Counsel, Autumn Venture - SC Ve... | About Standard CharteredWe are a leading inter... | 2022-04-01 | a1ad3581a81222507fa918dc2d978ed1db672c44415c3f... | 25fa191ddad0bb854bd7bbe811437b1c820271351c4ac8... | Autumn Life Pte. Ltd. | 0 | 0 | JobStreetSG | ['No Requirement'] |
| 3 | 0110a85844f5aa0b87060109b25567903d2188130391b2... | https://www.efinancialcareers.sg/jobs-Singapor... | Product/Data Analyst | About us<br/>Endowus is Asia's leading fee-onl... | 2022-04-01 | 4aac38458acbd96d2de7ceda69e0c3f9923c5c8b4773f5... | 3c7c5b38bc57d7f85029e41a219822866fe3dd6587027a... | Endowus |  |  |  |  |

The ones we will be using will be ****************************jobDescription**************************** and ****************jobTitle****************

## **Data Cleaning**

```python
def clean_text(text):
    # remove htmltags and new lines/tags
    try:
        text = re.sub(r'<.[a-zA-Z]+.>', ' ', text)
        text = re.sub(r'&.[a-zA-Z]+.;', '', text)
        #text = re.sub(r'^[a-zA-Z.]', '', text)
        text = re.sub(r'httpS+s*', ' ', text)
        text = re.sub(r'\.', '', text)
        text = re.sub(r'\(', '',text)
        text = re.sub(r'\)', '',text)
        text = re.sub(r' +', ' ',text)
        text = text.lower()
    except Exception as e:
        print(f"Error: {text}")
        return text

    return text
```

It is common understanding that preprocessing of texts data is very important in NLP tasks. However, with pretrained BERT frameworks, especially sentence-BERT, it is best that we do not perform common lexical preprocessing such as lemmatization, stemming, and stopword removals. BERT also has an internal tokenizer to process it. [See this article for details](https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/).

Here, we simply remove some level of “dirty” texts that do not contribute towards the semantic meaning of the sentence, such as extra spaces, links, html elements ie. *<\ br>* or *&amp;*

```python
# preprocess sample
clean_sent(clean_text(raw_data.jobDescription[68]))
```

> `'responsibilitiesmanage lead team provide residential services effectivelymaintain synergy hotel on-site managing agent resident councilresponsible time attendance record supervised team ensure accurate billing process mcstoversee maintenance accurate updated occupant records ensuring staff adherence confidentiality residents contact details personal informationconduct regular staff meetings maintain open channel communicationresolve resident complaints management office maintain high level resident satisfaction service qualitycommunicate on-site management office resident feedback recurring challenges improvementsparticipate site meetings convened management office requiredconduct participate yearly service excellence audit collaborate on-site managing agent meet compliancemaintain ongoing schedules ensure residential facilities safe clean attractivemaintain compliance regulatory requirements including workplace health amp safety occupational health amp safetyrequirementsdiploma hospitality management equivalentmin years experience least years leadership role luxury hospitality servicestrong leadership communication skillsability build trusting relationships stakeholdersproficient ms office-'`
> 

Sample job description after cleaning

```python
if raw_data.isnull().values.any():
    raw_data.dropna(how='any', inplace=True)
raw_data.reset_index(drop=True, inplace=True)
raw_data.set_axis(range(len(raw_data)), inplace=True)
```

We also remove any rows that contain `NaN` in its **jobDescription** and **********jobTitle********** columns

## Calculating **TF-IDF Matrix**

Here, we use Sklearn to produce the Dictionary as well as the tfidf_matrix.
For context,

- TF (Term Frequency): word occurence in a doc / total number of words in a doc, describing the rarity of a word
- IDF (Inverse Document Frequency): number of docs with appearance of given word / total number of documents, describing the frequency of word appearance.
- Each vector’s value is 0 - 1.0 . A higher value represent a higher importance and correlation, where 1.0 would mean an identical corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer="word")
bow_vectorizer = CountVectorizer(tokenizer=tokenize, analyzer="word")

corpus = resume.Resume_str
tfidf_matrix = vectorizer.fit_transform(corpus)
bow_matrix   = bow_vectorizer.fit_transform(corpus)
print(f"tfidf_matrix shape: {tfidf_matrix.shape}")
print(f"bow_matrix shape: {bow_matrix.shape}")
```

There are several available libraries that allow you to calculate TFIDF values quickly. The one I used in this case will be from **scikit-learn.**

`TfidfVectorizer` will first create a vocabulary of words by counting the occurence of words in a one-hot encoding vector, and then calculate its tfidf value based on it. [You can learn more about its inner working here](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

## **Cosine Similarity**

```markdown
## Cosine similarity matrix of a corpus
Cosine score is 0 (no similarity) and 1 (exact same)

${sim(A,B)}$ = ${\cos(\theta)}$ = ${ {A\cdot B} \over ||A||||B|| }$
```

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
bow_cosine_sim = cosine_similarity(bow_matrix, bow_matrix)t
print(cosine_sim)
print(f"cosine_sim shape: {cosine_sim.shape}")

cosine_sim_lk = linear_kernel(tfidf_matrix, tfidf_matrix)
```

Cosine similarity **measures the similarity between two vectors of an inner product space**.
 It is measured by the cosine of the angle between two vectors and 
determines whether two vectors are pointing in roughly the same 
direction. With this, we can estimate if the embedding vectors of two documents are similar, and rank them accordingly.

```python
tfidf_matrix shape: (2484, 32351)
bow_matrix shape: (2484, 32351)
[[1.         0.25575024 0.23616097 ... 0.09187018 0.10855762 0.09396372]
 [0.25575024 1.         0.20146724 ... 0.07162671 0.1392705  0.0711357 ]
 [0.23616097 0.20146724 1.         ... 0.0621186  0.09232892 0.08819255]
 ...
 [0.09187018 0.07162671 0.0621186  ... 1.         0.0711868  0.17859443]
 [0.10855762 0.1392705  0.09232892 ... 0.0711868  1.         0.06674726]
 [0.09396372 0.0711357  0.08819255 ... 0.17859443 0.06674726 1.        ]]
cosine_sim shape: (2484, 2484)
```

## Inverse Document Indexing

```python
from tqdm import tqdm

def inverted_index(words):
    """
        An ivnerted index of words (given word, find docID and idx)
    """
    inverted = {}
    for idx, word in enumerate(words):
        loc = inverted.setdefault(word, [])
        loc.append(idx)
    return inverted

def inverted_index_add(inverted, docID, doc_idx):
    for word in doc_idx.keys():
        loc = doc_idx[word]
        indices = inverted.setdefault(word, {})
        indices[docID] = loc
        
    return inverted

corpus = resume.Resume_str
inverted_doc_idx = {}
word_corpus = {}
with tqdm(total=len(corpus)) as pbar:
    for docid, x in enumerate(corpus):
        words = tokenize(x)
        word_corpus[docid] = words
        inv_idx = inverted_index(words)
        inverted_index_add(inverted_doc_idx, docid, inv_idx)
        pbar.update(1)
```

Finally, we generate an inverted index table. An inverted index lists every unique word that appears in any document and identifies all of the documents each word occurs in. This is similar, in a high level, to how elasticsearch performs it search.

## **Search Function**

```python
# Make sure you run the above to get your tfidf mat first as we refer it internally
def ranked_search(query, firstx=10):
    tokens = tokenize(query)
    query_weights = {}
    # get all the weights of the documents in which the term existed
    #get documents that matches the key
    for mapword, wmap in inverted_doc_idx.items():
        appear_in_docs = list(wmap.keys())

        if mapword in tokens:
            # print(f"looking via: {tokens}")
            # print(f"found {mapword} in tokens for docs {appear_in_docs}")
            for docid in appear_in_docs:
                wordidx = list(vectorizer.get_feature_names_out()).index(mapword)
                tfidfval = tfidf_matrix[docid,wordidx]
                # we add the value onto that doc id. The more words scored for that doc, the heavier the weight
                query_weights[docid] = query_weights.get(docid,0) + tfidfval
                
    query_weights = sorted(query_weights.items(), key=lambda x:x[1], reverse=True)[:firstx]
    result = []
    for (docid, tfidfval) in query_weights:
        data = {
                    'Relevance': round(tfidfval*100,2),
                    'ID': docid, 
                    'Resume_str': resume.Resume_str.iloc[docid], 
                    'Category': resume.Category[docid]
                }
        result.append(data)
    result = pd.DataFrame(result)
    return result
```

```python
%time ranked_search("HR")

Relevance 	ID 	Resume_str 	Category
0 	58.90 	4 	HR MANAGER Skill Highlights ... 	HR
1 	51.87 	101 	REGIONAL HR BUSINESS PARTNER Hu... 	HR
2 	51.10 	58 	HR CONSULTANT Summary C... 	HR
3 	49.76 	92 	GLOBAL HR MANAGER Summary ... 	HR
4 	46.55 	85 	SENIOR HR BUSINESS PARTNER ... 	HR
5 	46.51 	31 	HR GENERALIST Professional Prof... 	HR
6 	46.18 	68 	HR DIRECTOR Summary HR Prof... 	HR
7 	45.31 	69 	HR PROFESSIONAL Summary Dep... 	HR
8 	42.35 	88 	REGIONAL HR DEPUTY MANAGER Summ... 	HR
9 	42.09 	65 	HR CONSULTING Summary 7+ yea... 	HR
```

**Challenges**

Some of the challenges for this method is: 

- Scalability: Computing TFIDF Matrix can be computationally expensive as it grows. For ~4000 documents, it took about 7minutes to compute the TF-IDF matrix. Adding new documents into the matrix also requires recomputing the matrix or a [form of partial fitting like this repo suggest](https://github.com/idoshlomo/online_vectorizers)
- Lexical dependency: Even though we are deployed some semblence of semantic meaning into our embeddings, ultimately a vocabulary on the number of occurence of words are still somewhat lexical. That means it’s value will heavily be influenced by how its being preprocessed and the statistics of word appearances.

**What’s Next?**

We will look into S-BERT as a way for embedding instead of TF-IDF values for a semantic search system.