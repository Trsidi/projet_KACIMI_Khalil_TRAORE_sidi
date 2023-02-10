**Sommaire**

[I. Introduction](#_rk1qoh74neqr)

[II. Jeu de données et traitement](#_2pt8bj6o8c95)

[A. Étude univariée](#_ohveqpg5s30f)

[B. Étude bivariée](#_nuekrvov5k7n)

[III. Modélisation et analyse du phénomène](#_9qavb0rpo64h)

[A. 1ère phase de modélisation (avant features selection)](#_bojhyjfwed5g)

[B. 2ème phase de modélisation (après feature selection)](#_x2nq165m4b1f)

[IV. Conclusion et limite de l'approche](#_ozr8zn9dfkyr)

[V. Annexes](#_y8ukq4nviegd)

#

1.
# **Introduction**

Notre projet s'articule dans le cadre du cours Machine learning : SVM et Réseaux de neurones sous python. Afin d'évaluer notre degré de maîtrise de la démarche économétrique en matière de méthodes de _machine learning_, nous nous sommes lancés dans un projet _Kaggle_ portant sur la détermination des critères financiers pouvant expliquer la faillite, ou non, des entreprises.

La base est initialement composée de 95 variables décrivant le profil financier d'une entreprise. L'objectif va être de déterminer le meilleur modèle pouvant prédire la faillite.

L'intérêt de notre démarche est de prouver l'efficacité, ou non, des modèles de machine learning dans la captation du séisme économique que peut représenter la faillite.

Si nous prenons le temps de nous poser la question : **Qu'est-ce qui fait qu'une entreprise dépose le bilan ou non ?** L'on pourrait évoquer des défaillances de gestion, des décisions d'investissements mal menées ou encore une mauvaise anticipation des changements du secteur d'activité (Alcatel, Nokia, Kodak, …).

Nous espérons pouvoir creuser plus en profondeur l'utilité de certains ratios financiers d'une entreprise, est le levier, ou la massue, que cela peut constituer pour elle.

1.
# **Jeu de données et traitement**

Dans le cadre de notre projet, nous avons opté pour une base comportant 96 variables pour un total de 6 819 observations. Nous sommes donc face à une base que l'on peut qualifier de "_Huge_" (un nombre important de variables et d'observations).

Notre variable à expliquer, ou _target_, est "_Bankrupt?_", que l'on renommera _"Bankruptcy_". Il s'agit d'une variable binaire qui prend les modalités suivantes : 0 pour une entreprise en non-faillite ; 1 pour une entreprise en faillite.

À l'image des fraudes, la faillite d'une entreprise est un événement relativement rare, un fait illustré par le déséquilibre au niveau de la répartition de notre _target_ puisque l'on remarque que :

![](RackMultipart20230210-1-29rhbn_html_6d7352c88e3f8cc.png)

- La non-faillite, _False_, représente 96.77 % de nos entreprises.
- La faillite, _True_, représente 3.23 % de nos entreprises.

On anticipe alors un besoin de procéder à un rééquilibrage de notre jeu de données via une méthode de rééchantillonnage. Nous partirons sur de "_L'OverSampling_" via l'algorithme SMOTE (Synthetic Minority Over-sampling Technique).

Les 95 variables explicatives, ou _features_, sont composées de 2 variables qualitatives et 93 variables quantitatives. La majorité de ces variables représente des indicateurs financiers servant à mesurer la performance financière de l'entreprise, des informations utiles quand il s'agit de déterminer la probabilité de "_déposer le bilan_"pour une tierce entreprise.

**Première étape** de notre travail d'étude : se pencher sur les données et essayer d'en corriger les éventuels défauts pour assurer la fiabilité des résultats de notre partie "Modélisation et analyse du phénomène".

Par "défauts" l'on entend la présence de : "_Outliers_", valeurs extrêmes, valeurs manquantes ou encore des problèmes d'échelle.

Notre base comporte bien deux des défauts susmentionnés ("_Outliers_" et écart d'échelle). Des défauts que nous nous sommes empressés de corriger via l'écart interquartile pour les "_Outliers_" (on perd alors 549 observations au total) et la standardisation (centrage, soustraction de l'espérance, réduction, division par l'écart type) pour l'homogénéité de nos échelles.

**Seconde étape** , nous nous devons d'extraire un maximum d'informations de ces données. Or, l'on remarque que le nombre de variables n'aide pas pour la partie **statistiques descriptives**. Nous pensons alors judicieux d'entamer une pré-étape de "_Features selection_", afin de réduire le nombre de dimensions et faciliter l'analyse statistique (uni, bi et multivariée).

Les techniques de réduction utilisées sont au nombre de trois : une régression pénalisée de type _LASSO_ (extrêmement contraignante), un _SGDClassifier_ avec comme penalité une régression pénalisée type _Elastic-Net_ (compromis entre _RIDGE_ et _LASSO_), une sélection _RFECV_ (Recursive Feature Elimination, Cross-Validated). Nous avions également tenté une sélection via _Random Forest_, mais les contraintes de calcul ont fait que nous avons abandonné la méthode.

À la suite, nous obtenons trois bases : base\_lasso, base\_elasticnet et base\_RFECV avec respectivement 22, 10 et 14 variables (en plus de notre _target_).

Des résultats qui viennent en contre-sens de la gradation qui a motivé notre choix. Les degrés de flexibilité des méthodes ne sont pas illustrés par le nombre de variables retenues.

**Troisième étape** , nous analysons statistiquement nos bases et leurs variables respectives pour déterminer la viabilité statistiques de nos couples "_target_/_feature_". C'est ce que nous allons explorer dans la partie "Étude bivariée".

Pour la suite de notre étude, nous avons fait le choix de modéliser avec la **base LASSO**. C'est la base qui comporte le plus de variables, 22, exempts de multicolinéarité puisque _LASSO_tranche quand deux variables sont fortement corrélées entre elles.

1.
## Étude univariée

Nous nous sommes intéressés à notre _target_ "_Bankruptcy_". Comme mentionné précédemment, à l'origine, notre base était extrêmement déséquilibrée, via de "_l'OverSampling_" nous avons réussi a artificiellement monté le nombre d'entreprises en faillite pour atteindre une égalité en termes de représentativité (50 % d'observations pour chacune de nos modalités).

1.
## Étude bivariée

Après nettoyage de notre base, nous avons effectué une série de tests :

- ⍴, correlation, pour nos variables quantitatives.
- Khi², pour nos variables qualitatives.
- T-test, ou test de _Student_, pour nos couples hybrides.

Ici l'objectif est de procéder à une vérification statistique de la véracité de la présence de certaines variables dans notre jeu de données. Pour notre base _LASSO_, deux variables n'ont pas passé le T-test et ont donc été écartées de la base. Nous passons d'une base de 22 variables à une base de 20 variables.

Plus bas, nous avons la matrice des corrélations pour nos variables quantitatives. Cette sélection étant issue d'une régression pénalisée type _LASSO_, nous remarquons l'absence de corrélations extrêmes et en conséquence l'absence de multicolinéarité.

![](RackMultipart20230210-1-29rhbn_html_1cb635cc418ac510.png)

1.
# **Modélisation et analyse du phénomène**

1.
## 1ère p **hase de modélisation** (avant _features selection_)

Après la _"feature selection_", nous avons lancé nos algorithmes sur la base de données finale. Nous utilisons trois modèles de machine learning (_Random Forest_, _XGBoost_ et _SVM_) et deux modèles de réseaux de neurones (_Perceptron_ et _ANN_). Nous ajoutons à cela un modèle de régression logistique.

L'objectif est de déterminer le modèle le plus performant. Notre variable dépendante étant une variable binaire, nous avons utilisé des modèles de classification. Cette première phase de modélisation concerne 20 variables.

![](RackMultipart20230210-1-29rhbn_html_8d972d1df87659a2.png)

Afin de comparer nos modèles, nous avons effectué une cross-validation avec 10 classes, nous pouvons observer l'évolution des _accuracy_ sur le graphique ci-dessus.

L'algorithme _XGBoost_ et le _Random Forest_ offrent les meilleurs résultats.

Le _SVM linéaire_ et le _perceptron_ présentent les plus faibles capacités prédictives avec des _accuracy_ inférieures à celles du modèle de régression logistique. Toujours sur les SVM, le modèle non linéaire semble être le plus adapté à notre jeu de données. Ce dernier donne de meilleures prédictions que la régression logistique, mais reste assez loin derrière nos deux meilleurs algorithmes que sont _XGBoost_ et _Random Forest_.

Graphique: écart type

![](RackMultipart20230210-1-29rhbn_html_4351c9932d81587a.png)

Étant donné l'inconstance de certaines courbes, nous nous intéressons aux variations de _l'accuracy_ entre les différentes classes pour chacun de nos modèles. Pour ce faire, nous nous intéressons à l'écart type entre les classes avec le graphique ci-dessus. En effet, un écart type élevé indique un risque de **sur-apprentissage** sur l'échantillon d'apprentissage.

Le _Random Forest_ et _XGBoost_ ont le plus faible risque de sur-apprentissage, a contrario, le _Perceptron_ et le _SVM_ linéaire affichent les écarts types les plus élevés et donc la probabilité de sur-apprentissage est plus élevée pour ces deux algorithmes.

Graphique: Évolution de l'_accuracy_ avec l'_ANN_

![](RackMultipart20230210-1-29rhbn_html_2f6671070c3b2818.png)

Pour le modèle _ANN_ l'_accuracy_ est quasiment constante, mais elle est relativement faible. Il affiche l'_accuracy_ le plus faible avec une valeur de 0.51.

1.
## 2ème phase de modélisation (après _feature selection_)

_XGBoost_ et _Random Forest_ se présentent comme étant nos meilleurs modèles avec des _accuracy_ très élevées et un risque de sur-apprentissage faible par rapport à la majorité des modèles. Nous tentons néanmoins de tuner notre SVM non linéaire afin de voir s'il peut afficher de meilleurs résultats. Pour cela, nous nous servons du _GridSearch._ Malgré cela, la qualité de notre modèle ne s'est pas améliorée, elle s'est même dégradée, passant d'une _accuracy_ avoisinant les 0,75 à une _accuracy_ de 0,51.

Finalement, nos meilleurs modèles restent _Random Forest_ et _XGBoos_t. Ces derniers présentent des résultats similaires, comme peuvent en témoigner le graphique d'évolution des _accuracy_ et les matrices de confusion disponibles en annexe.

1.
# **Conclusion et limite de l'approche**

Au terme de notre étude, nous sommes arrivés aux conclusions suivantes :

- Les modèles ensemblistes sont les plus adaptés pour prédire la faillite des entreprises.
- Nos deux modèles _Random Forest_ et _XGBoost_ offrent les meilleures capacités prédictives avec un risque de sur-apprentissage relativement faible.
- Les réseaux de neurones ainsi que le modèle _SVM_ linéaire sont les moins adaptés à notre jeu de données.

Concernant le travail en lui-même, il est important de noter que l'apprentissage des réseaux de neurones n'a pas été développé tant que ça. Il aurait été plus judicieux d'optimiser le paramétrage de ces derniers afin d'espérer avoir de meilleurs résultats. Il convient néanmoins de préciser malgré tout que l'ensemble de nos résultats est valable dans les limites de notre échantillon ainsi que de celles des variables à notre disposition. La partie _Feature selection_ aurait pu être faite autrement. Nous avons pris un parti pris en optant pour les régressions pénalisées, mais une sélection en étudiant les corrélations était également envisageable. L'exploration des données nous semble améliorable d'un point de vue économétrique, mais aussi graphique (possibilité de retranscrire le lien entre nos variables).

1.
# Annexes

Annexe 1: matrice de confusion Random Forest

![](RackMultipart20230210-1-29rhbn_html_54c5befdaa3deeb6.png)

Annexe 2: matrice de confusion XG boost

![](RackMultipart20230210-1-29rhbn_html_54c5befdaa3deeb6.png)