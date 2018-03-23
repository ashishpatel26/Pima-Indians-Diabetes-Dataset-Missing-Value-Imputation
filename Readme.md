
# 1. Pima Indians Diabetes Dataset  

The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.

It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. The variable names are as follows:

0. Number of times pregnant.
1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
2. Diastolic blood pressure (mm Hg).
3. Triceps skinfold thickness (mm).
4. 2-Hour serum insulin (mu U/ml).
5. Body mass index (weight in kg/(height in m)^2).
6. Diabetes pedigree function.
7. Age (years).
8. Class variable (0 or 1).


```python
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import Imputer

def model_fit(dataset):
    values = dataset.values
    X = values[:,1:8]
    Y = values[:,8]
    lda = LinearDiscriminantAnalysis()
    kfold = KFold(n_splits=3, random_state=7)
    result = cross_val_score(lda, X, Y, cv = kfold, scoring="accuracy")
    print("Result of LDA:", result.mean())
```


```python
pima = pd.read_csv("pima-indians-diabetes.data.csv", header=None)
print(pima.shape)
print((pima[[1,2,3,4,5]] == 0).sum())
pima.describe()

```

    (768, 9)
    1      5
    2     35
    3    227
    4    374
    5     11
    dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 2.Mark Missing Values


```python
pima[[1,2,3,4,5]] =  pima[[1,2,3,4,5]].replace(0, np.NaN)
print(pima.isnull().sum())
```

    0      0
    1      5
    2     35
    3    227
    4    374
    5     11
    6      0
    7      0
    8      0
    dtype: int64
    


```python
pima.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>78.0</td>
      <td>50.0</td>
      <td>32.0</td>
      <td>88.0</td>
      <td>31.0</td>
      <td>0.248</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>197.0</td>
      <td>70.0</td>
      <td>45.0</td>
      <td>543.0</td>
      <td>30.5</td>
      <td>0.158</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>125.0</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.232</td>
      <td>54</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>110.0</td>
      <td>92.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.6</td>
      <td>0.191</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10</td>
      <td>168.0</td>
      <td>74.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>0.537</td>
      <td>34</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>139.0</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.1</td>
      <td>1.441</td>
      <td>57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>189.0</td>
      <td>60.0</td>
      <td>23.0</td>
      <td>846.0</td>
      <td>30.1</td>
      <td>0.398</td>
      <td>59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>166.0</td>
      <td>72.0</td>
      <td>19.0</td>
      <td>175.0</td>
      <td>25.8</td>
      <td>0.587</td>
      <td>51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7</td>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>0.484</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>118.0</td>
      <td>84.0</td>
      <td>47.0</td>
      <td>230.0</td>
      <td>45.8</td>
      <td>0.551</td>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>107.0</td>
      <td>74.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.6</td>
      <td>0.254</td>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>103.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>83.0</td>
      <td>43.3</td>
      <td>0.183</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>115.0</td>
      <td>70.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>34.6</td>
      <td>0.529</td>
      <td>32</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_fit(pima)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-42-dfae15f44b8b> in <module>()
    ----> 1 model_fit(pima)
    

    <ipython-input-38-f1bc7e6079bc> in model_fit(dataset)
         10     lda = LinearDiscriminantAnalysis()
         11     kfold = KFold(n_splits=3, random_state=7)
    ---> 12     result = cross_val_score(lda, X, Y, cv = kfold, scoring="accuracy")
         13     print("Result of LDA:", result.mean())
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch)
        340                                 n_jobs=n_jobs, verbose=verbose,
        341                                 fit_params=fit_params,
    --> 342                                 pre_dispatch=pre_dispatch)
        343     return cv_results['test_score']
        344 
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in cross_validate(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, return_train_score)
        204             fit_params, return_train_score=return_train_score,
        205             return_times=True)
    --> 206         for train, test in cv.split(X, y, groups))
        207 
        208     if return_train_score:
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self, iterable)
        777             # was dispatched. In particular this covers the edge
        778             # case of Parallel used with an exhausted iterator.
    --> 779             while self.dispatch_one_batch(iterator):
        780                 self._iterating = True
        781             else:
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\parallel.py in dispatch_one_batch(self, iterator)
        623                 return False
        624             else:
    --> 625                 self._dispatch(tasks)
        626                 return True
        627 
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\parallel.py in _dispatch(self, batch)
        586         dispatch_timestamp = time.time()
        587         cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
    --> 588         job = self._backend.apply_async(batch, callback=cb)
        589         self._jobs.append(job)
        590 
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\_parallel_backends.py in apply_async(self, func, callback)
        109     def apply_async(self, func, callback=None):
        110         """Schedule a func to be run"""
    --> 111         result = ImmediateResult(func)
        112         if callback:
        113             callback(result)
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\_parallel_backends.py in __init__(self, batch)
        330         # Don't delay the application, to avoid keeping the input
        331         # arguments in memory
    --> 332         self.results = batch()
        333 
        334     def get(self):
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):
    

    ~\Anaconda3\lib\site-packages\sklearn\externals\joblib\parallel.py in <listcomp>(.0)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):
    

    ~\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, error_score)
        456             estimator.fit(X_train, **fit_params)
        457         else:
    --> 458             estimator.fit(X_train, y_train, **fit_params)
        459 
        460     except Exception as e:
    

    ~\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py in fit(self, X, y)
        427             Target values.
        428         """
    --> 429         X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)
        430         self.classes_ = unique_labels(y)
        431 
    

    ~\Anaconda3\lib\site-packages\sklearn\utils\validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        571     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
        572                     ensure_2d, allow_nd, ensure_min_samples,
    --> 573                     ensure_min_features, warn_on_dtype, estimator)
        574     if multi_output:
        575         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
    

    ~\Anaconda3\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        451                              % (array.ndim, estimator_name))
        452         if force_all_finite:
    --> 453             _assert_all_finite(array)
        454 
        455     shape_repr = _shape_repr(array.shape)
    

    ~\Anaconda3\lib\site-packages\sklearn\utils\validation.py in _assert_all_finite(X)
         42             and not np.isfinite(X).all()):
         43         raise ValueError("Input contains NaN, infinity"
    ---> 44                          " or a value too large for %r." % X.dtype)
         45 
         46 
    

    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').


> We got a **error** in train the model Because of ***Missing Values contain in Dataset***

# 3.Remove Rows With Missing Values


```python
pima.shape
```




    (768, 9)




```python
import numpy
pima[[1,2,3,4,5]] = pima[[1,2,3,4,5]].replace(0, numpy.NaN)
pima.dropna(inplace=True)
```


```python
pima.shape
```




    (392, 9)




```python
model_fit(pima)
```

    Result of LDA: 0.7883734586024662
    

# 4.Impute Missing Values


```python
pima.shape
```




    (768, 9)



# 1.Impute value with ***mean()***


```python
pima[[1,2,3,4,5]] = pima[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
pima.fillna(pima.mean(), inplace=True)
```


```python
pima.shape
```




    (768, 9)




```python
pima.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.00000</td>
      <td>155.548223</td>
      <td>33.600000</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.00000</td>
      <td>155.548223</td>
      <td>26.600000</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.15342</td>
      <td>155.548223</td>
      <td>23.300000</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.000000</td>
      <td>23.00000</td>
      <td>94.000000</td>
      <td>28.100000</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.000000</td>
      <td>35.00000</td>
      <td>168.000000</td>
      <td>43.100000</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.15342</td>
      <td>155.548223</td>
      <td>25.600000</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>78.0</td>
      <td>50.000000</td>
      <td>32.00000</td>
      <td>88.000000</td>
      <td>31.000000</td>
      <td>0.248</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.15342</td>
      <td>155.548223</td>
      <td>35.300000</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>197.0</td>
      <td>70.000000</td>
      <td>45.00000</td>
      <td>543.000000</td>
      <td>30.500000</td>
      <td>0.158</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>125.0</td>
      <td>96.000000</td>
      <td>29.15342</td>
      <td>155.548223</td>
      <td>32.457464</td>
      <td>0.232</td>
      <td>54</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_fit(pima)
```

    Result of LDA: 0.7643229166666666
    

# 2.Impute value with ***Impute() Function***


```python
pima[[1,2,3,4,5]] = pima[[1,2,3,4,5]].replace(0, numpy.NaN)
```


```python
values = pima.values
X = values[:,1:8]
Y = values[:,8]
impute = Imputer()
transformated_values = impute.fit_transform(values)
print(numpy.isnan(transformated_values).sum())
lda = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(lda, transformated_values, y, cv=kfold, scoring='accuracy')
print("Result of LDA:", result.mean())
```

    0
    Result of LDA: 0.7734375
    

    C:\Users\ashish.patel\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    C:\Users\ashish.patel\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    C:\Users\ashish.patel\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    


```python
transformated_values = pd.DataFrame(transformated_values)
transformated_values.isnull().sum()

```




    0    0
    1    0
    2    0
    3    0
    4    0
    5    0
    6    0
    7    0
    8    0
    dtype: int64


