import logging
import sys
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def predict_user():
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        domain = request.form.get('domain').lower()
        subdomain = request.form.getlist('subdomain')
        subdomain = " ".join(subdomain)
        skills = request.form.getlist('skills')
        skills = " ".join(skills)
        dataframe = pd.read_csv("Features_Df.csv")
        to_find = len(dataframe)
        dataframe.loc[to_find] = [to_find, domain, subdomain, skills, f"{domain} {subdomain} {skills}"]
        count = CountVectorizer()
        score_df = pd.DataFrame()
        score_lists = []
        feature_cols = ['domains', 'subdomain', 'wanted_skills']
        feature_weights = [0.5, 0.25, 0.25]
        for index, (feature, weight) in enumerate(zip(feature_cols, feature_weights)):
            feat_matrix = count.fit_transform(dataframe[feature])
            feat_sim = cosine_similarity(feat_matrix, feat_matrix)
            score_feat = pd.Series(feat_sim[to_find]).sort_values(ascending=False)
            score_lists.append(score_feat)
            score_df = pd.concat(score_lists, axis=1)
        score_df["weighted_avg"] = score_df.mul(feature_weights).sum(axis=1)
        dubloo_score = score_df.weighted_avg.sort_values(ascending=False)
        rec_ids = list(dubloo_score.index)
        temp = dataframe.iloc[rec_ids][1:6]
        result = temp.iloc[:, 1:-1]
        return render_template('simple.html',  tables=[result.to_html(classes='data')], titles=result.columns.values)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
