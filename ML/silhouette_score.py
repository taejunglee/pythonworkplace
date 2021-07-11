# -*- coding: utf-8 -*-
"""silhouette_score.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ALxus-Ry-6p1SFgs_FN8PCiUc4RcdCHW
"""

def get_silhouette_score(df, models):
    for model in models:
        model.fit(df)

        df['cluster'] = model.labels_

        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(df)

        df['pca_x'] = pca_transformed[:, 0]
        df['pca_y'] = pca_transformed[:, 1]

        plt.scatter(x=df.loc[:, 'pca_x'], y=df.loc[:, 'pca_y'], c=df['cluster'])
        plt.figure()

        score_samples = silhouette_samples(df, df['cluster'])

        df['silhouette_coeff'] = score_samples

        average_score = silhouette_score(df, df['cluster'])
        print(f'{model} \n Silhouette Analysis Score: {average_score}')
        print('-------------------------------------------------------------------')