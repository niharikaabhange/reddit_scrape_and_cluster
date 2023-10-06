import praw
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import re
import spacy
import datetime
import time
import mysql.connector

import schedule
import sys

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import numpy as np
from threading import Thread
from time import sleep
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


import warnings
warnings.filterwarnings("ignore")

db_config = {
    	"host": "localhost",
    	"user": "root",
    	"password": "root",
    	"database": "lab4",
	}

connection = mysql.connector.connect(**db_config)			
					
if connection.is_connected():
	cursor = connection.cursor()
	table_query = f'''CREATE TABLE if not exists ReBot (
    			ID int auto_increment primary key,
    			CREATETIME datetime,
    			Domain varchar(30),
    			Post_title longtext,
    			Keywords longtext,
    			Post_body longtext,
    			Image_text longtext,
    			URL longtext,
    			Comments longtext,
    			Comment_keywords longtext,
    			Author	varchar(20));'''
    			
	cursor.execute(table_query)
	connection.commit()
# Reddit API credentials
CLIENT_ID = 'cfp3bRBdcyqTmlS6SuVpVQ'
CLIENT_SECRET = 'oiKewjqT4GBPO9L-Pw2HWnjtICjVKQ'
USER_AGENT = 'python:restobotteam:v1 (by u/restobotteam)'

domains_keywords = {
    'Technology': ['technology', 'tech', 'gadgets', 'innovation', 'electronics', 'software', 'hardware', 'internet', 'digital', 'robotics'],
    'Healthcare': ['health', 'medical', 'wellness', 'biotech', 'medicine', 'pharmaceuticals', 'care', 'doctor', 'patient', 'nurse'],
    'Environment': ['environment', 'sustainability', 'green', 'ecology', 'climate', 'conservation', 'renewable', 'ecosystem', 'nature', 'carbon'],
    'Finance': ['finance', 'financial', 'banking', 'investment', 'economics', 'stocks', 'trading', 'money', 'economy', 'market'],
    'ArtificialIntelligence': ['ai', 'artificial', 'machine', 'deep', 'neural networks', 'algorithms', 'automation', 'intelligence', 'cognition', 'smart'],
    'Education': ['education', 'learning', 'teaching', 'school', 'knowledge', 'educational', 'study', 'student', 'academics', 'training'],
    'Space': ['space', 'universe', 'cosmos', 'astronomy', 'planets', 'galaxies', 'celestial', 'stars', 'exploration', 'orbit'],
    'Automotive': ['automotive', 'cars', 'vehicles', 'automobiles', 'auto', 'driving', 'car', 'driver', 'engine', 'transportation'],
    'Entertainment': ['entertainment', 'media', 'movies', 'music', 'games', 'gaming', 'video', 'film', 'art', 'show'],
    'Security': ['security', 'safety', 'protection', 'privacy', 'secure', 'defense', 'safeguard', 'guard', 'safety', 'shield']
}



# Function to preprocess the text
def preprocess_text(text):
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase and strip leading/trailing whitespaces
    text = text.lower().strip()
    
    return text

# Function to fetch top comments for a post
def fetch_top_comments(submission, num_comments=10):
    comments = []
    submission.comments.replace_more(limit=0)
    
    for comment in submission.comments[:num_comments]:
        comments.append(preprocess_text(comment.body))
    
    return comments

# Function to extract text from images using OCR (pytesseract)
def extract_text_from_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {str(e)}")
        return ""

# Function to fetch posts with pagination
def fetch_reddit_posts(subreddit, num_posts):
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    subreddit = reddit.subreddit(subreddit)

    # Initialize an empty list to store posts
    posts = []

    # Define the batch size for fetching posts
    batch_size = 100  # Adjust based on API rate limits

    # Calculate the number of batches needed to fetch the desired number of posts
    num_batches = -(-num_posts // batch_size)  # Ceiling division

    # Fetch posts in batches
    for i in range(num_batches):
        # Fetch a batch of posts
        batch = subreddit.new(limit=batch_size, params={'after': None if i == 0 else posts[-1]['url']})

        # Extend the list of posts with the current batch
        for submission in batch:
            post_title = preprocess_text(submission.title)
            post_body = preprocess_text(submission.selftext)

            # Extracting text from images in the post
            image_text = ""
            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                image_text = extract_text_from_image(submission.url)
                image_text = preprocess_text(image_text)

            # Fetch top comments
            comments = fetch_top_comments(submission)
            standard_datetime = datetime.datetime.utcfromtimestamp(submission.created_utc)
            posts.append({
                'post_title': post_title,
                'post_body': post_body,
                'image_text': image_text,
                'url': submission.url,
                'created_utc': standard_datetime,
                'author': 'Restonymous',
                'comments': comments
            })

        # Respect API rate limits (approximately 2 requests per second for most endpoints)
        time.sleep(0.5)  # Adjust as needed to stay within rate limits

    return pd.DataFrame(posts[:num_posts])  # Return the specified number of posts
	   
def categorize_domain(keywords, domain_keywords):
    for domain, keywords_list in domain_keywords.items():
        if any(keyword in keywords for keyword in keywords_list):
            return domain
    return 'Other'

  

def storage(newdf):
     if connection.is_connected():
     	cursor = connection.cursor()
     	insert_query = f"INSERT INTO ReBot ( CREATETIME, Domain, Post_title, Keywords, Post_body, Image_text, URL, Comments, Comment_keywords, Author) VALUES (%s, %s, %s, %s, %s, %s, %s,%s, %s, %s);"
     	for index, row in newdf.iterrows():
     		row['keywords'] = ', '.join(row['keywords']) if isinstance(row['keywords'], list) else row['keywords']
     		row['comments'] = ', '.join(row['comments']) if isinstance(row['comments'], list) else row['comments']
     		flattened_comment_keywords = ', '.join(', '.join(sublist) for sublist in row['comment_keywords']) if isinstance(row['comment_keywords'], list) else row['comment_keywords']
     		row['comment_keywords'] = flattened_comment_keywords
     		row_tuple = tuple(row)
     		cursor.execute(insert_query, row_tuple)
     	connection.commit()
     	print('\n---Updating database---\nResolving duplication...')
     	drop_duplicates_query = "DELETE r1.* FROM ReBot r1 JOIN ReBot r2 ON r1.id > r2.id AND r1.Post_title = r2.Post_title;"
     	cursor.execute(drop_duplicates_query)
     	connection.commit()
     	print('\n---Updating database---\nSent to server :)')


def extract_keywords(text):
    
    	nlp = spacy.load("en_core_web_sm")
    	doc = nlp(text)
    	unique_keywords = set()
    	for token in doc:
    		if not token.is_stop and token.is_alpha:
    			unique_keywords.add(token.text)
    	return list(unique_keywords)
    	
def extract_cluster_keywords(texty):
	nlp = spacy.load("en_core_web_sm")
	# Define the sentence you want to extract keywords from
	
	# Process the sentence with spaCy
	doc = nlp(texty)
	# Extract important keywords (nouns and adjectives) from the sentence
	keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]
	# Count the frequency of each keyword
	keyword_counts = Counter(keywords)
	# Extract keywords that occur more than once and are therefore important
	top_10_keywords = [keyword for keyword, count in keyword_counts.most_common(5)]
	# Print the important keywords
	return top_10_keywords

def preprocess_reddit(df):
    
    df['keywords'] = df.apply(lambda row: extract_keywords(row['post_title'] + ' ' + row['post_body'] + ' ' + row['image_text']), axis=1)
    df['domain'] = df.apply(lambda row: categorize_domain(row['keywords'],domains_keywords), axis=1)
    df['comment_keywords'] = df['comments'].apply(lambda comments: [extract_keywords(comment) for comment in comments])
    
    newdf = df[[ 'created_utc', 'domain', 'post_title', 'keywords', 'post_body', 'image_text', 'url', 'comments', 'comment_keywords', 'author']]
    return(newdf)

def plot_clusters(df, predicted_cluster, kmeans_model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the clusters
    for cluster in range(len(kmeans_model.cluster_centers_)):
        cluster_data = df[df['cluster_no'] == cluster]
        ax.scatter(cluster_data['X'], cluster_data['Y'], cluster_data['Z'], label=f'Cluster {cluster}')

    # Plot the predicted point
    #predicted_cluster_data = df[df['cluster_no'] == predicted_cluster]
    #ax.scatter(predicted_cluster_data['X'], predicted_cluster_data['Y'], predicted_cluster_data['Z'], c='r', label='Predicted Point')
    #ax.scatter(predicted_cluster[0], predicted_cluster[1], predicted_cluster[2], c='r', label='Predicted Point')

    plt.title('Clustering Visualization')
    plt.legend()
    plt.show()  
    
def plot_clusters_2d(df, predictioned):
    plt.figure()

    # Plot the clusters
    for cluster in df['cluster_no'].unique():
        cluster_data = df[df['cluster_no'] == cluster]
        plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster}')

    # Plot the centroids
    plt.scatter(predictioned[0][0], predictioned[0][1], c='r', marker='X', label='Predicted')

    plt.title('Clustering Visualization (2D PCA)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
def remove_stop_words(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence


def vectorizer_trainer():
	if connection.is_connected():
		cursor = connection.cursor()
		select_query = "SELECT Post_title, Comments FROM ReBot;"
		cursor.execute(select_query)
		result = cursor.fetchall()
		concatenated_list = [(''.join(tpl)) for tpl in result]
		tagged_data = [TaggedDocument(words=word_tokenize(remove_stop_words(doc.lower())), tags=[str(i)]) for i, doc in enumerate(concatenated_list)]
		model = Doc2Vec(vector_size=20,min_count=2, epochs=50)
		model.build_vocab(tagged_data)
		model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
			
	return concatenated_list, model
		
		
def vectorizer(data,model):
	new_vector = model.infer_vector(word_tokenize(remove_stop_words(data.lower())))
	
	return new_vector


def cluster_trainer(model, concatenated_list):
	a =  model.docvecs.get_normed_vectors()
	
	NUMBER_OF_CLUSTERS = 10
	kmeans_model = KMeans(n_clusters=NUMBER_OF_CLUSTERS)
	centroids = kmeans_model.fit(a).cluster_centers_
	doc_ids = kmeans_model.labels_
	# computes cluster Id for document vectors
	pca = PCA(n_components=2)
	a_pca = pca.fit_transform(a)
	
	mydata = {
	    'sentence' : concatenated_list,
	    'cluster_no' : doc_ids,
	    'Centroid_x' : centroids[doc_ids, 0],
	    'Centroid_y' : centroids[doc_ids, 1],
	    'X': 0,
	    'Y': 0
	}
	df = pd.DataFrame(mydata)
	for i in range(0,len(a_pca)):
		df['X'][i] = a_pca[i][0]
		df['Y'][i] = a_pca[i][1]
	print(df,'\n\n')
	return df, kmeans_model,pca,centroids
	
	
def clusterizer(new_vector, kmeans_model):
	predicted_cluster = kmeans_model.predict(new_vector.reshape(1, -1))[0]
	return predicted_cluster
	
	
	

def handle_user_input():
		
		user_input = input("\nEnter text to cluster or 'quit' to stop: ")
		if user_input.lower() == 'quit':
			return 1
		else:
			# TRAINING
			print('\n---Training Model---\nVectorizing training data')
			concatenated_list, model = vectorizer_trainer()
			print('\n---Training Model---\nClustering training data')
			df, kmeans_model,pca, centroids = cluster_trainer(model, concatenated_list)
			#print(df)
			all_clusters = []
			cluster_keywords = []
			for cluster_num in range (0, 10):
			
				currdf = df[df['cluster_no'] == cluster_num]
				startstr = ''
				for sen in currdf['sentence']:
					startstr = startstr + sen
				all_clusters.append(startstr)
				cluster_keywords.append(extract_cluster_keywords(startstr))
					
			#print('\n\nTraining data is clustered into', len(all_clusters), 'clusters', cluster_keywords)
			# PREDICTING NEW VALUE
			print('\n---Clustering User Input---\nUser input vectorizing')
			new_vector = vectorizer(user_input,model)
			print('\n---Clustering User Input---\nUser input clustering')
			predicted_cluster = clusterizer(new_vector, kmeans_model)
			print('\n---Results---\nPrediction for ', user_input, ' is ', predicted_cluster)
			print('The keywords of this cluster are', cluster_keywords[predicted_cluster])
			print(new_vector)
			new_vector_2d = new_vector.reshape(1, -1)
			transformed_vector = pca.transform(new_vector_2d)
			print(transformed_vector)
			
			plot_clusters_2d(df,transformed_vector)
			return 0

def update_database(update_interval):
	while True:
		print('\n---Updating database---\nFetching new posts...')
		df = fetch_reddit_posts(subreddit, num_posts)
		print('\n---Updating database---\nPreprocessing and storing fetched data...')
		newdf = preprocess_reddit(df)
		storage(newdf)
		
		sleep(update_interval)


if __name__ == "__main__":
	print("entered main")
	subreddit = 'tech'
	num_posts = int(input("Enter number of posts to fetch : "))
	update_interval = int(sys.argv[1])
	daemon = Thread(target=update_database, args=(update_interval*60,), daemon=True, name='Background')
	daemon.start()
	print(f"\nFetching script will run every {update_interval} minute(s).")
	while True:
		i = handle_user_input()
		if i==1:
			break
		else:
			time.sleep(1)
			
			
# visualize clusters
# clear the database once			
# 			
			
			
