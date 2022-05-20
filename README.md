# EDA Notebook

**All the methods used in the project are documented in the notebook.**
**Therefore, in this documentation I will talk about the general ideas and the workflow of the program.**

The process of data analysis and dataviz it's pretty repetitive. In order to organize the project better, I created methods for all kind of things. All the functions can be found in the EDA notebook. Now, I will explain the main ideas for each dataframe (`articles`, `customers`, `transactions`).

## `Articles`

The dataframe contains a lot of pairs of type (*categorical*, *numerical*). For data processing, the *categorical* attributes are more useful for understanding the data, but some ML algorithms suffer from *categorical* features. In this case, we can use the *numerical* variables, or encode the *categorical* ones.

Keypoints from `articles` dataframe:
- Only the *detail_desc* field has some missing values, which is good
- H&M has a lot of *ladieswear* and very little *sportwear* :(
- There are a lot of *upper body* and *lower body* garment in the store
- The main articles are *black*, *dark blue* and *white* 

## `Customers`

This dataframe contains details about each customer. 

Keypoints from `customers` dataframe:
- H&M has about 1.3mil customers during the sampling period
- *FN* and *Active* has a lot of missing values (>60%). Therefore, we can't use these features for training a model. If we had a small portion of missing values, we could use an *Imputer* with different strategies (depending on the type of each attribute) to complete the missing values
- The distribution of ages is not a *Gaussian* one, but if we look at it within the intervals $[10-40]$ and $[40-80]$, we can compare them with a *Normal* distribution. The common ages are 20-25 in the first frame and 50-55 in the second one
- The ages are *right skewed*. We have a lot of young guys over there :)
- A lot of customers prefer not to get any news from H&M


## `Transactions`

This dataframe contains details about each transaction: what article is sold, to whom, for what price and through which channel

Keypoints from `transactions` dataframe:
- H&M has about 30mil transactions captured in this dataset
- The sampling was done in the period 2018-09-20 - 2020-09-22 (2 years)
- Most sales are done through the *second channel*
- The *price* attribute has a lot of outliers, especially high prices
- Most sales are done in the month *July* and *Saturday* (when grouping the transactions by the month of the year)
- Looking at evolution of the price in time for some categories, we can clearly see a price increase in *October 2019*
- Another interesting fact about the given data is that it satisfies the *Pareto Principle* (the graph can be found in the notebook)

# ML Notebook

**For this part of the project, I tried 2 different approaches**

- `Only dataframe manipulations`
- `ALS model`

## 1. Only dataframe manipulations

**I implemented a recommendation system using only the existing dataframes, without involving a ML algorithm. I used this type of algorithm because it's very intuitive and doesn't require any ML knowledge, it's a good way of understanding the data better.**

The process of recommendation items to customers is the following:

- First, recommend items `previously purchased` by the current user
- Then, items that are `bought together` with previous purchases
- At the end, we can also recommend `top N popular items`

The initial research was done on the following notebooks:

- https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635

        This trick is very useful for large dataframes. You can drastically reduce the memory of the dataframe fitting in memory.
        The main idea is to convert the df columns to some another type which requires fewer bits to store in the memory. Another trick is to do a 1:1 mapping for `customer_ids` (the process hold in both directions, because the mapping function is bijective).

- https://www.kaggle.com/code/hengzheng/time-is-our-best-friend-v2/notebook

        This guy had a good idea to use only the last part of the dataset, instead of trying to recommend items based on the entire period of 2 years.
        Therefore, I used only a last fraction of the dataset (the last 2 weeks, more exactly).

- https://www.kaggle.com/code/cdeotte/customers-who-bought-this-frequently-buy-this/notebook

        The idea to recommend items which are frequently purchased together is present in this notebook. Knowing this information, we can predict which items a customer will buy after we observe what they have already bought. 


## 2. Alternating Least Squares model

**This approach is based on a matrix factorization technique used in collaborative filtering. ALS is a matrix factorization algorithm and it runs itself in a parallel fashion. PySpark module has the implementation of the ALS algorithm with all the functionalities, but it doesn't have support for GPU. Therefore, I used an open-source library called [`implicit`](https://github.com/benfred/implicit/).**

The way this algorithm works is the following:

- The input matrix $R$ of type [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) or [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) and with the format (`users` x `items`) is given to the ALS algorithm

- The algorithm factorizes the matrix $R$ into 2 factors $U$ and $V$, such that $R \approx U^TV$. Since matrix factorization can be used in the context of recommendation, the matrices $U$ and $V$ can be called `user` and `item` matrix, respectively. The matrix $R$ can be called the `ratings` matrix and is given in its sparse representation as a tuple of $(i,j,r)$ where $i$ denotes the row index, $j$ the column index and $r$ is the matrix value at position $(i, j)$.

- In order to find the user and item matrix, the following problem is solved:
$$\arg\min_{U,V} \sum_{\{i,j\mid r_{i,j} \not= 0\}} \left(r_{i,j} - u_{i}^Tv_{j}\right)^2 +
\lambda \left(\sum_{i} n_{u_i} \left\lVert u_i \right\rVert^2 + \sum_{j} n_{v_j} \left\lVert v_j \right\rVert^2 \right)$$

- This regularization scheme (with $\lambda$ being the *regularization factor*) is used to avoid overfitting and is called weighted-$\lambda$-regularization.

- By fixing one of the matrices $U$ or $V$, we obtain a quadratic form which can be solved directly. The solution of the modified problem is guaranteed to monotonically decrease the overall cost function. By applying this step alternately to the matrices $U$ and $V$, we can iteratively improve the matrix factorization.


The initial research was done on the following notebooks:

- https://www.kaggle.com/code/nadianizam/h-m-fashion-recommendation-with-pyspark

        I used the idea to implement a recommender system with the ALS algorithm. The data manipulations are also very useful and interesting to look at.

- https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014

        In this notebook I saw that a good metric for validating the model could be `mean average precision at K` (MAP@K).
        Another keypoint is the auto-incrementing indices starting from 0 to both users and items. Doing this, we can easily compute the input matrix of type (users x items) for fitting the ALS model.
        I've also made a grid search for tuning the hyperparameters and plot the learning curves to see how the model performs when is fitted with more and more data. 


## Comparison between the 2 approach variants

**The first approach is a more data-scientist version, because it doesn't use any ML algorithms. It's based only on dataframes operations and basic items recommendation ideas.**

**Second approach is more stable and it's used a lot in recommender systems, especially after the [Netflix Prize](https://www.researchgate.net/publication/220788980_Large-Scale_Parallel_Collaborative_Filtering_for_the_Netflix_Prize) contest. It's also more scalabe than the dataframe manipulations and it can be applied to any type of [implicit data](https://www.youtube.com/watch?v=fdWcq7Sdpf4).**

**It may seem surprising, but the best personal leaderboard score (0.0208) was obtained with the first approach.**


## 3 challenges

- One difficulty that I had was with the [RAPIDS cudf](https://github.com/rapidsai/cudf) library. The API has a special way of loading the dataframes and sometimes CUDA throws an error saying it's out of memory.

        I solved this problem using the classical Pandas Dataframe: I loaded the tables (dfs) using the Pandas API and then convert to cudf. The purpose of using RAPIDS cudf was to have GPU support with df operations. Another interesting thing that I've done was to reduce the memory of the dataframes by re-mapping the existing types to ones with less bits representation (the transformation is bijective, so I can reverse the process whenever is needed).

- Another interesting challenge that I had was the misunderstanding of `implicit` package version. The latest stable version is `0.5.0` and has some important changes in it.

        With older versions, you needed to provide the training matrix in the form `items x users`, but after the breaking change API you need to provide a csr matrix of form `users x items`
        Another interesting thing happend when I was using the model to `recommend` items to users. Older versions of the API don't allow you to provide a list of `user_ids` to the recommender system. To overcome this challenge I've decided to use a version greater than 0.5.0. The recommendation step is faster than predicting items for each user sequentially.

- The process of fine-tuning the model was also interesting, because the `implicit` library doesn't have support for `Grid Search` with the `ALS` model.

        For solving this problem, I used a class called `ParameterGrid` from `sklearn`. It allows you to provide a dictionary of type {'hyperparam_name': [list_of_values]}. It's a more elegant way of doing `for in fors` and it's more scalable (with adding/removing parameters or values in/from the dictionary)
        In the same manner, I implemented a function for plotting the `learning_curves` of the model, starting with little training data and feeding the model with more and more training instances.

## 3 things that I learned / improved

- Improved the EDA part with df manipulations, plots, charts and interpretations of the results. Learned that Pandas is way slower than RAPIDS when talking about dataframe manipulations (e.g: grouping, counting-by, filtering)

- Learned how the recommendation system works and the under the hood math explanations

- Learned that the data processing step is very important. In this manner you "help" the model to understand the data better and generalize well on future instances.

## 3 ideas for further improving the solution

- Maybe try a `PCA` followed by a `TSNE` or another dimensionality-reduction technique to find some interesting clusters of users. After that, we can take each user (customer) and find its appropriate cluster (group). Then, we can recommend items frequently buyed in that group.

- Unsupervised algorithms may be useful here. `K-Means` would be a great candidate. Maybe it will find some good clusters.

- A Neural Network may do some great work, because we are dealing with a dataframe with a lot of transactions and it's hard to come with a good solution using some simple algorithms.
