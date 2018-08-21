# Recommendation Engines

<div style="margin-top: 10%">
</div>

## Problem statement:

Suggest **items** (product, movie, song,...) that a **user** would want to buy.

---

## Feature types

| Type | Example | Quality | Volume |
| --- | --- | --- | --- |
| Explicit (user gave rating)| Star ratings on Amazon | High | Low |
| Implicit (infer rating from behavior)| Browsing history |Low | High |

<div style="margin-top:1%">
</div>

* **Explicit ratings**:

  Feedback from the user about how much they liked the item.
* **Implicit ratings**:

  Infer feedback from user by whether or not they used or browsed the item.

---
## Different types of recommenders:

* **Content-based recommendation:** (e.g. Pandora)
  * Items are mapped (usually by hand) into a feature space.
  * Build a _profile_ for each user's preferences in feature space.
  * Recommendations: find "similar" item.
  * Examples: label movies/songs/books by genre
* **Collaborative recommendation:** (e.g. Amazon)
  * Builds a _profile_ for users **and** items to a feature space.
  * Recommendations: find items liked by similar users to current user.
  * Has "cold start" problem for new users / items.
* **Hybrid:** combination of collaborative and content-based (e.g. Netflix)
---

## Content-based filtering


### Ratings matrix

<table class="recommend">
<tr>
  <th></th>
  <th>The Matrix</th>
  <th>Notebook</th>
  <th>Incredibles</th>
  <th>Shawshank</th>
  <th>Forrest Gump</th>
</tr>
<tr>
  <th>U0: Athena</th>
  <td>4.0</td>
  <td class="unknown">?</td>
  <td>3.5</td>
  <td>4.0</td>
  <td class="unknown">?</td>
</tr>
<tr>
  <th>U1: Sam</th>
  <td class="unknown">?</td>
  <td>1.0</td>
  <td>3.0</td>
  <td class="unknown">?</td>
  <td>2.0</td>
</tr>
<tr>
  <th>U2: Andy</th>
  <td>2.0</td>
  <td class="unknown">?</td>
  <td class="unknown">?</td>
  <td>4.0</td>
  <td class="unknown">?</td>
</tr>
<tr>
  <th>U3: Lindsey</th>
  <td class="unknown">?</td>
  <td>5.0</td>
  <td class="unknown">?</td>
  <td class="unknown">?</td>
  <td>3.0</td>
</tr>
</table>

<div style="align:center; margin-top: 50px;">
These contain the *known* ratings of user $i$ on movie $j$.
</div>
---

## Content-based filtering

* **MANUALLY** Map each item $j$ to a feature space:
  <div class='highlight-box'>
  e.g. movie data set might have
  `$$M[j,:] = \left[m_{j0}, m_{j1},  m_{j2}, ..., m_{jN}\right]$$`
  where `$m_{j0}$` is "amount of action in movie $j$",<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  `$m_{j1}$` is "amount of comedy in movie $j$", et cetra.
  </div>
* For each user, **FIND** (using ML) a $N$-dimensional feature vector $\theta_i$ that describes how much the user likes each of the features.
  <div class='highlight-box'>
  e.g. movie data set might have
  `$$\Theta[i,:] = \left[\theta_{i0}, \theta_{i1}, \ldots, \theta_{iN}\right]$$`
  where `$\theta_{i0}$` is "amount that user $i$ likes action", <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  `$\theta_{i1}$` is "amount user $i$ likes comedy", et cetera.
  </div>

---

## Content-based filtering (continued)

Let $M$ be the movie matrix (given):
<table class="recommend">
  <tr>
    <th></th>
    <th>X0: Action</th>
    <th>X1: Comedy</th>
    <th>X2: Romance</th>
  </tr>
  <tr>
    <th>M0: The Matrix</th>
    <td>1.0</td>
    <td>0</td>
    <td>0.3</td>
  </tr>
  <tr>
    <th>M1: The Notebook</th>
    <td>0.0</td>
    <td>0.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <th>M2: The Incredibles</th>
    <td>0.6</td>
    <td>0.3</td>
    <td>0.05</td>
  </tr>
  <tr>
    <th>M3: Shawshank Redemption</th>
    <td>1.0</td>
    <td>0.2</td>
    <td>0.1</td>
  </tr>
  <tr>
    <th>M4: Forrest Gump</th>
    <td>0.2</td>
    <td>0.8</td>
    <td>0.6</td>
  </tr>
</table>

Say we know Athena's vector (we will see how to construct it later):
$$\Theta[0, :] = [ 0.163, -0.721,  0.034]$$
and her average movie rating is a 3.666. How do we get her ratings?

---

### Predicting Athena's ratings:

<div class='highlight-box'>
$$ \text{predicted ratings for user $i$} = \mu_i + \Theta[i, :] * M^T $$


<div style="margin: auto 0; text-align:center">(Matrix multiplication, not element-wise)
</div>
</div>

For Athena (note movie matrix is transposed):
`$$\text{prediction} = 3.666 + \left[0.163, -0.721, -0.034\right]
\left(
\begin{array}{ccc}
1.0 & 0.0 & 0.6 & 1.0 & 0.2\\
0.0 & 0.0 & 0.3 & 0.1 & 0.8\\
0.3 & 1.0 & 0.05 & 0.1 & 0.6
\end{array}
\right)$$`

`$$\phantom{prediction} = 3.666 + \left[0.173, 0.034, -0.117, 0.022, -0.524\right]$$`

`$$\phantom{prediction} = \left[4.00 , 3.87  , 3.72 , 3.86, 3.31 \right]$$`


<div class='highlight-box'>
`$$ \text{Athena's actual ratings} = \left[4.0, ?, 3.5, 4.0, ?\right] $$`
i.e. we would predict Athena would rate the Notebook at 3.87, and Forrest Gump at 3.31
</div>

---

### Calculating the error:

<table class="recommend">
  <tr>
    <th></th>
    <th>The matrix</th>
    <th>The notebook</th>
    <th>The incredibles</th>
    <th>Shawshank</th>
    <th>Forrest Gump</th>
  </tr>
  <tr>
    <th>Atheta Predicted</th>
    <td>4.0</td>
    <td>3.87</td>
    <td>3.72</td>
    <td>3.86</td>
    <td>3.31</td>
  </tr>
  <tr>
    <th>Atheta Actual</th>
    <td>4.0</td>
    <td class="unknown"> ?</td>
    <td>3.5</td>
    <td>4.0</td>
    <td class='unknown'>?</td>
  </tr>
  <hr>
  <tr>
    <th>Error</th>
    <td>0.0</td>
    <td class='unknown'></td>
    <td>0.22</td>
    <td>-0.14</td>
    <td class='unknown'></td>
  </tr>
</table>
&nbsp;
<div class="align:center; margin-top: 40px">
Sum of squared errors: $(0)^2 + (0.22)^2 + (-0.14)^2 = 0.068$
</div>

&nbsp;

<div class="align:center; margin-top: 40px">
Note we only calculate the error for known ratings. This is what we want to minimize.
</div>

---

### Problem rephrased:

* For each user $i$, there are some ratings on movies that we know. Let `$\vec{y} = r_{ij}$`
* For each user $i$, there is an average rating
* For each movie, we know the features $M$
* We want to use the ratings that we know parameters $\Theta[i,:]$ to find the best fit line to
`$$ \vec{y} = \mu_i + \Theta[i, :] M^T$$`
on the known movies. Once we have this, use the $\Theta[i,:]$ to predict for the unknown movies.



We have seen this problem before!

---
### Problem rephrased (cont):

i.e. This is linear regression, with a different set of coefficients $\Theta[i,:]$ for each user.
- The movies that have ratings are the ones we train on, and get errors for
- The movies without ratings are the ones we predict.

* We generally include regularization:

<div class='highlight-box'>
  For each user, find $\Theta[i,:]$ to minimize the regularized cost
  `$$J(\Theta[i,:]) = \sum_{j\text{ rated}} (\mu_i + \Theta[i, :] M^T - r_{ij})^2 + \lambda |\Theta[i,:]|^2$$`
</div>

---

### Finding Athena's vector

So far, we assumed that we had Athena's "parameter" vector $\theta$. To find it:
```python
> M = np.array([[1.0, 0.0, 0.3],
                [0.0, 0.0, 1.0],
                [0.6, 0.3, 0.05],
                [1.0, 0.2, 0.1],
                [0.2, 0.8, 0.6]])
> athena_seen=[0, 2, 3]
> athena_rating=np.array([4.0, 3.5, 4.0])
> athena_demean_rating=athena_rating - athena_rating.mean()
> lr = Ridge(intercept=False, alpha=0.05)  
> lr.fit(M[athena_seen, :], athena_demean_rating)
# Get Athena's "theta" parameter
> lr.coef_
array([ 0.16298422, -0.72101772,  0.03378626])
# Get predictions for Athena on all movies:
> lr.predict(M) + athena_rating.mean()
array([4.00645343, 3.86711959, 3.71650786, 3.85549264, 3.30938775])
```
---
### Everyone's theta vector

- Note that the different "rows" of the user matrix don't mix: we can calculate each user in parallel.
- Relies entirely on good labelling of features!
- There is an algebraic solution as well, as we can rewrite our cost function for each user as
`$$J(\Theta[i,:]) = ||(\Theta[i,:]M^T + \mu_i - R[i,:])Si||^2 + \lambda ||\Theta[i,:]||^2$$`
where $Si$ is a $n_m \times n_m$ diagonal matrix so that `$Si[j,j]$`  is 1 if user $i$ has seen movie $j$ and 0 otherwise. This trick allows us to set the unseen ratings to zero without having them contribute to the error.

If we define $r$ as the matrix with the mean subtracted, the solution is
<div class='highlight-box'>
`$$\Theta[i,:] = ((Si M)^T (Si M) + \lambda 1)^{-1} (Si M)^T r$$`
</div>
---

### See it in code

Let's see it in code to predict Athena's vector (recall all this mess is to stop us from manually splitting training data from prediction data)

```python
# M is still our movie matrix from last time
> R = np.array([[4.    , np.nan, 3.5   , 4.    , np.nan],
                [np.nan, 1.    , 3.    , np.nan, 2.    ],
                [2.    , np.nan, np.nan, 4.    , np.nan],
                [np.nan, 5.    , np.nan, np.nan, 3.    ]])
> is_seen = ~np.isnan(R)
# Look at just athena, user 0
> S0 = np.diag(is_seen[0])
> r_demean = R - np.nanmean(R, axis=1).reshape(-1,1)
> r_demean[~is_seen] = 0   # fill nan with 0
> SM0 = S0 @ M
> Theta0 = np.linalg.inv(SM0.T @ SM0 + 0.05*np.identity(3)) @ SM0.T @ r_demean[0]
array([ 0.16298422, -0.72101772,  0.03378626])
```
While a little more opaque, this means we don't have to manually select the movies Athena has seen.

---

### For another user

The strength of this is when trying to find the parameters for many users. Here is the additional work we would need to find $\theta$ for user 1 (Sam)
```python
> S1 = np.diag(is_seen[1])
> SM1= S1 @ M
> Theta1 = np.linalg.inv(SM1.T @ SM1 + 0.05*np.identity(3)) @ SM1.T @ r_demean[1]
array([ 1.34610204,  0.40559406, -0.95765265])
```
This is (arguably) easier than having to separate out manually which movies each user watched, and training a ridge regression on each one.

<P>
(If you thought the previous method was clearer, I am inclined to agree. We are putting this in here as a secret weapon for later.)

---

### Summary of content based:

- Requires someone to give the $N$ item features (in this case, $M$)
- Use Linear Regression to find $N$ user features from the known ratings
- Then use LR to predict new ratings, recommend highest rated films.
- Once user vector is known, rating is highest for movies $\theta$ pointing in a similar direction to user vector $\theta$. Can use cosine similarity to find vectors pointing in similar directions.
- Will generally recommend the same "type" or "genre" -- hard to explore or be surprised.

---

## Collaborative filtering

---

## Collaborative filtering: overview

We will be looking at _neighborhood_ models
* **EITHIER**:
  * Find **users** "similar to" user X, and weight recommendation scores based on what "similar users" like.
  * Find **items** "similar to" item Y, and weight recommendation scores based on how user rated "similar items".
* Does not require manual labeling of user features or item features.
* Does require large overlap between user and items, hard for new users / items (the "cold start" problem)
* Generally item-item preferred (number of items << number of users)

---

## Collaborative filtering

<table class="recommend">
<tr>
  <th></th>
  <th>I1</th>
  <th>I2</th>
  <th>I3</th>
  <th>I4</th>
  <th>I5</th>
  <th>I6</th>
  <th>I7</th>
  <th>I8</th>
  <th>I9</th>
  <th>I10</th>
  <th>I11</th>
  <th>I12</th>
</tr>
<tr>
  <th>U1</th>
  <td>1</td>
  <td></td>
  <td>3</td>
  <td></td>
  <td></td>
  <td>5</td>
  <td></td>
  <td></td>
  <td>5</td>
  <td></td>
  <td>4</td>
  <td></td>
</tr>
<tr>
  <th>U2</th>
  <td></td>
  <td></td>
  <td>5</td>
  <td>4</td>
  <td></td>
  <td></td>
  <td>4</td>
  <td></td>
  <td></td>
  <td>2</td>
  <td>1</td>
  <td>3</td>
</tr>
<tr>
  <th>U3</th>
  <td>2</td>
  <td>4</td>
  <td></td>
  <td>1</td>
  <td>2</td>
  <td></td>
  <td>3</td>
  <td></td>
  <td>4</td>
  <td>3</td>
  <td>5</td>
  <td></td>
</tr>
<tr>
  <th>U4</th>
  <td></td>
  <td>2</td>
  <td>4</td>
  <td></td>
  <td>5</td>
  <td></td>
  <td></td>
  <td>4</td>
  <td></td>
  <td></td>
  <td>2</td>
  <td></td>
</tr>
<tr>
  <th>U5</th>
  <td></td>
  <td></td>
  <td>4</td>
  <td>3</td>
  <td>4</td>
  <td>2</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td>2</td>
  <td>5</td>
</tr>
<tr>
  <th>U6</th>
  <td>1</td>
  <td></td>
  <td>3</td>
  <td></td>
  <td>3</td>
  <td></td>
  <td></td>
  <td>2</td>
  <td></td>
  <td></td>
  <td>4</td>
  <td></td>
</tr>
</table>

We will implement user-user similarity. Let's say we want similarities for user1 (U1). Who is most similar?

---

### Cosine similarity

* Experiments show standard scaling doesn't work, but mean norm does

|  User |  U1 | U2 | U3 | U4 | U5 | U6 |
| --- | --- | --- | --- | --- | --- | --- |
| Mean | 3.6 | 3.1667 | 3 | 3.4 | 3.333 | 2.6|


* Subtract means and  `nan`s with 0 after subtraction:
<table class="recommend">
<tr>
  <th></th>
  <th>I1</th>
  <th>I2</th>
  <th>I3</th>
  <th>I4</th>
  <th>I5</th>
  <th>I6</th>
  <th>I7</th>
  <th>I8</th>
  <th>I9</th>
  <th>I10</th>
  <th>I11</th>
  <th>I12</th>
</tr>
<tr>
  <th>U1</th>
  <td>-2.6</td>
  <td>0</td>
  <td>-0.6</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
</tr>
<tr>
  <th>U2</th>
  <td>0</td>
  <td>0</td>
  <td>1.833</td>
  <td>0.833</td>
  <td>0</td>
  <td>0</td>
  <td>0.833</td>
  <td>0</td>
  <td>0</td>
  <td>-1.167</td>
  <td>-2.167</td>
  <td>-0.167</td>
</tr>
<tr>
  <th>U3</th>
  <td>-1</td>
  <td>1</td>
  <td>0</td>
  <td>-2</td>
  <td>-1</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>1</td>
  <td>0</td>
  <td>2</td>
  <td>0</td>
</tr>
<tr>
  <th>U4</th>
  <td>0</td>
  <td>-1.4</td>
  <td>0.6</td>
  <td>0</td>
  <td>1.6</td>
  <td>0</td>
  <td>0</td>
  <td>0.6</td>
  <td>0</td>
  <td>0</td>
  <td>-1.4</td>
  <td>0</td>
</tr>
<tr>
  <th>U5</th>
  <td>0</td>
  <td>0</td>
  <td>0.667</td>
  <td>-0.33</td>
  <td>0.667</td>
  <td>-1.33</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>-1.33</td>
  <td>1.67</td>
</tr>
<tr>
  <th>U6</th>
  <td>-1.6</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
  <td>0</td>
  <td>-0.6</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
</tr>
</table>

---

### Cosine similarity (cont)

Find the cosine between the each user and user1:
$$\cos\theta = \frac{u_1 \cdot u_i}{|u_1| |u_i|}$$
<table class="recommend">
<tr>
  <th></th>
  <th>I1</th>
  <th>I2</th>
  <th>I3</th>
  <th>I4</th>
  <th>I5</th>
  <th>I6</th>
  <th>I7</th>
  <th>I8</th>
  <th>I9</th>
  <th>I10</th>
  <th>I11</th>
  <th>I12</th>
  <th>cos&theta;</th>
</tr>
<tr>
  <th>U1</th>
  <td>-2.6</td>
  <td>0</td>
  <td>-0.6</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
  <td><b>1.0</b></td>
</tr>
<tr>
  <th>U2</th>
  <td>0</td>
  <td>0</td>
  <td>1.833</td>
  <td>0.833</td>
  <td>0</td>
  <td>0</td>
  <td>0.833</td>
  <td>0</td>
  <td>0</td>
  <td>-1.167</td>
  <td>-2.167</td>
  <td>-0.167</td>
  <td><b>-0.179</b></td>
</tr>
<tr>
  <th>U3</th>
  <td>-1</td>
  <td>1</td>
  <td>0</td>
  <td>-2</td>
  <td>-1</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>1</td>
  <td>0</td>
  <td>2</td>
  <td>0</td>
  <td><b>0.414</b></td>
</tr>
<tr>
  <th>U4</th>
  <td>0</td>
  <td>-1.4</td>
  <td>0.6</td>
  <td>0</td>
  <td>1.6</td>
  <td>0</td>
  <td>0</td>
  <td>0.6</td>
  <td>0</td>
  <td>0</td>
  <td>-1.4</td>
  <td>0</td>
  <td><b>-0.102</b></td>
</tr>
<tr>
  <th>U5</th>
  <td>0</td>
  <td>0</td>
  <td>0.667</td>
  <td>-0.33</td>
  <td>0.667</td>
  <td>-1.33</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>0</td>
  <td>-1.33</td>
  <td>1.67</td>
  <td><b>-0.309</b></td>
</tr>
<tr>
  <th>U6</th>
  <td>-1.6</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
  <td>0.4</td>
  <td>0</td>
  <td>0</td>
  <td>-0.6</td>
  <td>0</td>
  <td>0</td>
  <td>1.4</td>
  <td>0</td>
  <td><b>0.587</b></td>
</tr>
</table>

---

### Cosine similarities: getting the neighborhood $U$

i.e. User 1 is most similar to U6 (cosine sim = 0.587) and U3 (cosine sim = 0.414). We can choose to ....

* Keep the most similar $k$ users (e.g. $k=2$ for user 1 would pick U3 and U6)
* Keep users with similarity above threshold (i.e. only keep strong signals)
* Whether or not to keep strong negative correlations (sometimes they help, other times they hurt.)

When making a recommendation for user 1 on item $j$, the users that meet our cutoff and have purchased item $j$ are called the _neighborhood_ `$N_{U_1}$` of user 1 and item $j$

---

### User-User collaborative recommendation:

Our estimated recommendation is
<div class='highlight-box'>
`$$ r_{ij} = \mu_i +  \frac{\sum_{u \in N_{U_1}} (r_{uj} - \mu_u) w_u}{\sum_{u\in N_{U_1}}|w_u|}$$`
</div>
where the sum is over users in the neighborhood `$N_{U_1}$`.

#### Notes:
* We are still interested weighting the deviation from the mean (mean normalization)
* It is OK to fill in nulls with 0 for calculating cosine similarity, as 0s are effectively "dropped" from the calculation. It **isn't** OK to do this before mean normalization (you will lower the mean artificially).

---
### Recommendation for item 5 for user 1

* Neighborhood is U3 and U6
* Information:

| User | Cosine sim with U1 | Average rating | Rating on item 5 | Rating above ave |
| --- | --- | --- | --- | --- |
| U1 | 1.0 | 3.6 | ??? | ??? |
| U6 | 0.587 | 2.6 | 3 | 0.4 |
| U3 | 0.414 | 3 | 2 | -1.0 |

&nbsp;
<div class="margin-top:15px">
</div>
#### Calculation:
`$$r_{1,5} = 3.6 + \frac{(0.4)*0.587 + (-1.0)*0.414}{0.587+0.414} = 3.6 - 0.18 = 3.42$$`


---
### Cold start

* Note how we needed to have _similar_ users that have rated the _item_ before we can recommend it.
* At the moment, no good way of recommending item 12 to user 1 (neither U3 nor U6 have rated it).
* Problem for new users!
* Often items are introduced less frequently, and have more people rate them. Item-item collaborative filtering (getting dot product between item vectors) more common.

---

### A different take on collaborative filtering

Can take approach similar to content based filtering, where we try to assign both user and item features. Write the ratings matrix $R$ as
$$ R = U \times M^T $$
where
- $U$ is a $n_u \times d$ matrix, where $d$ is number of new features
- $M$ is a `$n_{items} \times d$` matrix.

Here $d$ plays a role similar to number of genres in content-based filtering.

* If $R$ were completely known, this would be a matrix factorization problem (see **SVD**)
* $R$ has missing values (the ones we want to predict) and we generally want to regularize.

---

### Regularization and missing data make things complicated

* Generally need to implement SGD on dataset, as linear algebra doesn't easily cope with missing values.
* Can write in Keras or TensorFlow. Math isn't bad, but doesn't scale well in realistic systems.
* Can also add neighborhood techniques (helps with accuracy and scaling)
* An alternative approach is ALS (Alternating Least Squares)

---

### ALS

Basic algorithm:
<div style='font-size: 24pt;'>
<ol>
<li> Start with a randomly initialized $M$ matrix.</li>
<li> **Treat $M$ as fixed.** For each user find exact solution for $\Theta$ using this $M$:
`$$\Theta[i,:] = ((Si M)^T (Si M) + \lambda 1)^{-1} (Si M)^T r[i, :]$$`
Recall $Si$ is the diagonal _item_ $\times$ _item_ matrix that "masks" the rated items for us, and $r$ are the demeaned ratings. Do this for all users.</li>
<li> **Now treat $\Theta$ as fixed.** Find the exact $M$ for this $\Theta$ using the algebraic technique.
`$$M[j,:] = ( ((Sj \Theta)^T (Sj \Theta) + \lambda 1)^{-1} (Sj \Theta)^T r[:, j] ).T$$`
Here $Sj$ is the diagonal _user_ $\times$ _user_ matrix that "masks" which users have rated item $j$. Do this for all items.</li>
<li> Take your newly computed $M$ and go back to step 2 to update $\Theta$. Then in step 3, update you $M$. Keep **alternating** between solving for $M$ and $\Theta$ until you converge.</li>
</ol>
</div>
---

### Further results
