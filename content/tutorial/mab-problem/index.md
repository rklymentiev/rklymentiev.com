---
title: "How to Deal with Multi-Armed Bandit Problem"
date: "2019-04-01"
summary: How to get maximum performance of your A/B test when you have any prior information about users.
image:
  caption: 'Image by <a href="https://pixabay.com/users/AidanHowe-15857243/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5012428">Aidan Howe</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5012428">Pixabay</a>'
  focal_point: ""
  preview_only: false
  
authors:
  - admin

tags:
  - Reinforcement Learning
  - Python
  - Python
---

*by Ruslan Klymentiev and [Zank Bennett](https://github.com/zankbennett)*

*Originally created for Bennett Data Science [blog post](https://bennettdatascience.com/know-your-options-when-it-comes-to-a-b-testing/).*

## What is Multi-Armed Bandit Problem?

This is a classic case that comes up various areas, from marketing to medical trials: How do you give a user what they want before you have a relationship with them? What about the case where you don’t even know much about the affinity for a user to an item or treatment your giving the user, as in a medical trial of several drugs or supplements?

We’ll show you how to approach these problems, and how a Multi-Armed Bandit with Thompson Sampling is generally the best choice. That’s a lot of words. Let’s slow down and start with something simple.

As an example, imagine you are running a marketing campaign for your website and have two ads to choose from. And say the objective is  to show the ad with highest click through rate (CTR) value to drive the highest traffic possible. But you don't have any prior information about how either of these ads will perform. What would be your approach? What about typical A/B testing? How about showing both of the ads equally, then, at some point, switching to the ad with the highest measured CTR? How long do you have to show both ads before settling on a “winner”? In these cases, it seems like guessing might be our best bet. In fact, that’s not far off!


There’s a method for this very problem; it’s called **the multi armed bandit (MAB)**.

There’s a lot of information explaining what [the MAB algorithm](https://en.wikipedia.org/wiki/Multi-armed_bandit) is, what it does and how it works, so we’ll keep it brief. Essentially, the MAB algorithm is a near-optimal method for solving the explore exploit trade-off dilemma that occurs when we don’t know whether to explore possible options to find the one with the best payoff or exploit an option that we feel is the best, from the limited exploration done so far.

In this tutorial we will look at different algorithms to solve the MAB problem. They all have different approaches and different Exploration vs Exploitation ratios.

Here, we’ll define **CTR** as the ratio of how many times an ad was clicked vs. the number of impressions. For example, if an ad has been shown 100 times and clicked 10 times, CTR = 10/100 = 0.1

We’ll define **regret** as the difference between the highest possible CTR and the CTR shown. For example, if ad A has a known CTR or 0.1 and ad B has a known CTR of 0.3, each time we show ad A, we have a regret equal to 0.3 - 0.1 = 0.2. This seems like a small difference until we consider that an ad may be shown 1MM times in only hours.

In this tutorial, we want to demonstrate which algorithm implementation performs best in terms of minimizing regret. The four implementations we’ll use are:

1. Random Selection
2. Epsilon Greedy
3. Thompson Sampling
4. Upper Confidence Bound (UCB1)

Each method will be described below in the simulations.

To perform this experiment, we have to assume we know the CTR’s in advance. That way, we can simulate a click (or not) of a given ad. For example, if we show ad A, with a known CTR of 28%, we can assume the ad will be clicked on 28% of the time and bake that into our simulation.


```python
import numpy as np
import pandas as pd
from scipy.stats import beta, bernoulli
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import random
import math
```

<details><summary>Custom functions</summary>
<p>

```python
def algorithm_performance(chosen_ads, total_reward, regret_list):
    """
    Function that will show the performance of each algorithm we will be using in this tutorial.
    """

    # calculate how many time each Ad has been choosen
    count_series = pd.Series(chosen_ads).value_counts(normalize=True)
    print('Ad A has been shown', count_series[0]*100, '% of the time.')
    print('Ad B has been shown', count_series[1]*100, '% of the time.')

    print('Total Reward (Number of Clicks):', total_reward) # print total Reward

    x = np.arange(0, n, 1)

    # plot the calculated CTR for Ad A
    trace0 = go.Scatter(x=x,
                       y=ctr['A'],
                       name='Calculated CTR for Ad A',
                       line=dict(color=('rgba(10, 108, 94, .7)'),
                                 width=2))

    # plot the line with actual CTR for Ad A
    trace1 = go.Scatter(x=[0, n],
                       y=[ACTUAL_CTR['A']] * 2,
                       name='Actual CTR for Ad A',
                       line = dict(color = ('rgb(205, 12, 24)'),
                                   width = 1,
                                   dash = 'dash'))

    # plot the calculated CTR for Ad B
    trace2 = go.Scatter(x=x,
                       y=ctr['B'],
                       name='Calculated CTR for Ad B',
                       line=dict(color=('rgba(187, 121, 24, .7)'),
                                 width=2))

    # plot the line with actual CTR for Ad A
    trace3 = go.Scatter(x=[0, n],
                       y=[ACTUAL_CTR['B']] * 2,
                       name='Actual CTR for Ad B',
                       line = dict(color = ('rgb(205, 12, 24)'),
                                   width = 1,
                                   dash = 'dash'))

    # plot the Regret values as a function of trial number
    trace4 = go.Scatter(x=x,
                       y=regret_list,
                       name='Regret')

    fig = make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=True)

    fig.add_trace(trace0, row=1, col=1)
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace4, row=2, col=1)

    fig.update_layout(
        title='Simulated CTR Values and Algorithm Regret',
        xaxis={'title': 'Trial Number'},
        yaxis1={'title': 'CTR value'},
        yaxis2={'title': 'Regret Value'}
    )
    
    return fig
```

</p>
</details>
<br>

This never(?) happens in real life, but for we will assume that we know the actual CTR values for both Ads for simulation purposes.  


```python
ads = ['A', 'B']
ACTUAL_CTR = {'A': .45, 'B': .65}
```

## Random Selection

* *0% - Exploration*
* *100% - Exploitation*

Let's start with the most naïve approach - Random Selection. The Random Selection algorithm doesn't do any Exploration, it just chooses randomly the Ad to show. 

You can think of it as coin flip - if you get heads you show Ad A, if you get tails you show Ad B. So if you have 2 ads, each add will appear ~50% (=100%/2) of the time. I guess you can tell already what are the disadvantages of this model, but let's look on simulation.


```python
# For each alrorithm we will perform 1000 trials
n = 1000
```


```python
regret = 0 
total_reward = 0
regret_list = [] # list for collecting the regret values for each impression (trial)
ctr = {'A': [], 'B': []} # lists for collecting the calculated CTR 
chosen_ads = [] # list for collecting the number of randomly choosen Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0}
clicks = {'A': 0, 'B': 0}

for i in range(n):    
    
    random_ad = np.random.choice(ads, p=[1/2, 1/2]) # randomly choose the ad
    chosen_ads.append(random_ad) # add the value to list
    
    impressions[random_ad] += 1 # add 1 impression value for the choosen Ad
    did_click = bernoulli.rvs(ACTUAL_CTR[random_ad]) # simulate if the person clicked on the ad usind Actual CTR value
    
    if did_click:
        clicks[random_ad] += did_click # if person clicked add 1 click value for the choosen Ad
    
    # calculate the CTR values and add them to list
    if impressions['A'] == 0:
        ctr_0 = 0
    else:
        ctr_0 = clicks['A']/impressions['A']
        
    if impressions['B'] == 0:
        ctr_1 = 0
    else:
        ctr_1 = clicks['B']/impressions['B']
        
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    
    # calculate the regret and reward
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[random_ad]
    regret_list.append(regret)
    total_reward += did_click
```


```python
fig = algorithm_performance(chosen_ads, total_reward, regret_list)
fig.show()
```


<iframe src="/mab-problem/1.html" width="700" height="450" frameborder="0"></iframe>


Both Ads were shown equal amount of times and the more trials, the closer the CTR values are to their known values. However, the Regret is continually increasing since the algorithm doesn't learn anything and doesn't do any updates according to gained information. This ever-increasing regret is exactly what we’re hoping to minimize with “smarter” methods.

I would use this algorithm in two cases:

1. I want to be confident about the estimated CTR value for each Ad (the more impression each Ad get, the more confindet I am that estimated CTR equals to real CTR).

2. I have unlimited Marketing Budget ;)


```python
# save the reward and regret values for future comparison
random_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)
}
```

## Epsilon Greedy

* *~15% - Exploration*
* *~85% - Exploitation*

The next approach is Epsilon-Greedy algorithm. Its logic can be defined as follows:
1. Run the experiment for some initial number of times (**Exploration**).
2. Choose the winning variant with the highest score for initial period of exploration.
3. Set the Epsilon value, **$\epsilon$**.
4. Run experiment with choosing the winning variant for **$(1-\epsilon)\% $** of the time and other options for **$\epsilon\%$** of the time (**Exploitation**).

Let's take a look at this algorithm in practice:


```python
e = .05 # set the Epsilon value
n_init = 100 # number of impressions to choose the winning Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0}
clicks = {'A': 0, 'B': 0}

for i in range(n_init):
    random_ad = np.random.choice(ads, p=[1/2, 1/2]) # randomly choose the ad
    
    impressions[random_ad] += 1
    did_click = bernoulli.rvs(ACTUAL_CTR[random_ad])
    if did_click:
        clicks[random_ad] += did_click
        
ctr_0 = clicks['A'] / impressions['A']
ctr_1 = clicks['B'] / impressions['B']
win_index = np.argmax([ctr_0, ctr_1]) # select the Ad number with the highest CTR

print('After', n_init, 'initial trials Ad', ads[win_index], 
      'got the highest CTR =', round(np.max([ctr_0, ctr_1]), 2), 
      '(Real CTR value is', ACTUAL_CTR[ads[win_index]], ').'
      '\nIt will be shown', (1-e)*100, '% of the time.')
```


```python
regret = 0 
total_reward = 0
regret_list = [] # list for collecting the regret values for each impression (trial)
ctr = {'A': [], 'B': []} # lists for collecting the calculated CTR 
chosen_ads = [] # list for collecting the number of randomly choosen Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0}
clicks = {'A': 0, 'B': 0}

# update probabilities
p = [e, e]
p[win_index] = 1 - e

for i in range(n):    
    
    win_ad = np.random.choice(ads, p=p) # randomly choose the ad
    chosen_ads.append(win_ad) # add the value to list
    
    impressions[win_ad] += 1 # add 1 impression value for the choosen Ad
    did_click = bernoulli.rvs(ACTUAL_CTR[win_ad]) # simulate if the person clicked on the ad usind Actual CTR value
    
    if did_click:
        clicks[win_ad] += did_click # if person clicked add 1 click value for the choosen Ad
    
    # calculate the CTR values and add them to list
    if impressions['A'] == 0:
        ctr_0 = 0
    else:
        ctr_0 = clicks['A']/impressions['A']
        
    if impressions['B'] == 0:
        ctr_1 = 0
    else:
        ctr_1 = clicks['B']/impressions['B']
        
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    
    # calculate the regret and reward
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
    regret_list.append(regret)
    total_reward += did_click
```


```python
fig = algorithm_performance(chosen_ads, total_reward, regret_list)
fig.show()
```

<iframe src="/mab-problem/2.html" width="700" height="450" width="100%" frameborder="0"></iframe>


That’s much better; Notice how the regret has decreased by an order of magnitude! The Epsilon-Greedy algorithm seems to perform much better than Random Selection. But can you see the problem here? The winning variant from exploration period will not necessarily be the optimal variant, and you can actually exploit the suboptimal variant. This increases regret and decreases reward. According to the **Law of Large Numbers** the more initial trials you do, the more likely you will choose the winning variant correctly. But in Marketing you don't usually want to rely on chance and hope that you have reached that 'large number of trials'.

> In probability theory, the **law of large numbers (LLN)** is a theorem that describes the result of performing the same experiment a large number of times. According to the law, the average of the results obtained from a large number of trials should be close to the expected value, and will tend to become closer as more trials are performed.

The good point is that you can adjust the ratio that controls how often to show the winning ad after initial trials by choosing different $\epsilon$ values. 


```python
epsilon_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}
```

## Thompson Sampling 

* *50% - Exploration*
* *50% - Exploitation*

The Thompson Sampling exploration part is more sophisticated than e-Greedy algorithm. We have been using **Beta distribution**\* here, however Thompson Sampling can be generalized to sample from any distributions over parameters.

> \*In probability theory and statistics, the **beta distribution** is a family of continuous probability distributions defined on the interval [0, 1] parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$, that appear as exponents of the random variable and control the shape of the distribution. 

*If you want to know more about Beta distribution here is an [article](http://varianceexplained.org/statistics/beta_distribution_and_baseball/) I found extremely useful.*

Logic:

1. Choose prior distributions for parameters $\alpha$ and $\beta$.
2. Calculate the $\alpha$ and $\beta$ values as: $\alpha = prior + hits$, $\beta = prior + misses$. * In our case hits = number of clicks, misses = number of impressions without a click. Priors are useful if you have some prior information about the actual CTR’s of your ads. Here, we do not, so we’ll use 1.0.*
3. Estimate actual CTR’s by sampling values of Beta distribution for each variant $B(\alpha_i, \beta_i)$ and choose the sample with the highest value (estimated CTR).
4. Repeat 2-3.

<details><summary>**Custom functions**</summary>
<p>

```python
# functions for manual Tompson sampling

def calculate_beta_dist(win_ad):
    impressions[win_ad] += 1
    did_click = bernoulli.rvs(ACTUAL_CTR[win_ad])
    if did_click:
        clicks[win_ad] += did_click

    # update ctr values according to beta destribution expected values
    ctr_0 = random.betavariate(priors['A']+clicks['A'], priors['A'] + impressions['A'] - clicks['A'])
    ctr_1 = random.betavariate(priors['B']+clicks['B'], priors['B'] + impressions['B'] - clicks['B'])
    highest_ad = np.argmax([ctr_0, ctr_1])
    chosen_ads.append(highest_ad)

    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    return highest_ad


def plot_beta_dist():
    x = np.arange(0, 1, 0.01)
    y = beta.pdf(x, priors['A']+clicks['A'], priors['B'] + impressions['A'] - clicks['A'])
    y /= y.max() ## normalize

    trace0 = go.Scatter(x=x,
                    y=y,
                    name='Beta Distribution (Ad A)',
                    marker = dict(color=('rgba(10, 108, 94, 1)')),
                    fill='tozeroy',
                    fillcolor = 'rgba(10, 108, 94, .7)')

    trace1 = go.Scatter(x = [ACTUAL_CTR[0]] * 2,
                    y = [0, 1],
                    name = 'Actual CTR A Value',
                    mode='lines',
                    line = dict(
                        color = ('rgb(205, 12, 24)'),
                        width = 2,
                        dash = 'dash'))

    y = beta.pdf(x, priors['A']+clicks['B'], priors['B'] + impressions['B'] - clicks['B'])
    y /= y.max()

    trace2 = go.Scatter(x=x,
                    y=y,
                    name='Beta Distribution (Ad B)',
                    marker = dict(color=('rgba(187, 121, 24, 1)')),
                    fill='tozeroy',
                    fillcolor = 'rgba(187, 121, 24, .7)')

    trace3 = go.Scatter(x = [ACTUAL_CTR[1]] * 2,
                    y = [0, 1],
                    name = 'Actual CTR B Value',
                    mode='lines',
                    line = dict(
                        color = ('rgb(205, 12, 24)'),
                        width = 2,
                        dash = 'dash'))

    fig = go.Figure([data1, data2, data3, data4])
    fig.updatedate_layout(
        title='Beta Distributions for both Ads',
        xaxis={'title': 'Possible CTR values'},
        yaxis={'title': 'Probability Density'})

    fig.show()
```
</p>
</details>

```python
regret = 0 
total_reward = 0
regret_list = [] 
ctr = {'A': [], 'B': []}
index_list = [] 
impressions = {'A': 0, 'B': 0}
clicks = {'A': 0, 'B': 0}
priors = {'A': 1, 'B': 1}

win_ad = np.random.choice(ads, p=[1/2, 1/2]) ## randomly choose the first shown Ad

for i in range(n):    
    
    impressions[win_ad] += 1
    did_click = bernoulli.rvs(ACTUAL_CTR[win_ad])
    if did_click:
        clicks[win_ad] += did_click
    
    ctr_0 = random.betavariate(priors['A']+clicks['A'], priors['B'] + impressions['A'] - clicks['A'])
    ctr_1 = random.betavariate(priors['A']+clicks['B'], priors['B'] + impressions['B'] - clicks['B'])
    win_ad = ads[np.argmax([ctr_0, ctr_1])]
    chosen_ads.append(win_ad)
    
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
    regret_list.append(regret)    
    total_reward += did_click
```

<details><summary>Code</summary>
<p>

```python
## plot the Beta distributions
x = np.linspace(0,1,1000)
y = beta.pdf(x, priors['A']+clicks['A'], priors['B'] + impressions['A'] - clicks['A'])
y /= y.max() ## normalize

trace0 = go.Scatter(x=x,
                   y=y,
                   name='Beta Distribution (Ad A)',
                   marker = dict(color=('rgba(10, 108, 94, 1)')),
                   fill='tozeroy',
                   fillcolor = 'rgba(10, 108, 94, .7)')

trace1 = go.Scatter(x = [ACTUAL_CTR['A']] * 2,
                   y = [0, 1],
                   name = 'Actual CTR A Value',
                   mode='lines',
                   line = dict(
                       color = ('rgb(205, 12, 24)'),
                       width = 2,
                       dash = 'dash'))

y = beta.pdf(x, priors['A']+clicks['B'], priors['B'] + impressions['B'] - clicks['B'])
y /= y.max()

trace2 = go.Scatter(x=x,
                   y=y,
                   name='Beta Distribution (Ad B)',
                   marker = dict(color=('rgba(187, 121, 24, 1)')),
                   fill='tozeroy',
                   fillcolor = 'rgba(187, 121, 24, .7)')

trace3 = go.Scatter(x = [ACTUAL_CTR['B']] * 2,
                   y = [0, 1],
                   name = 'Actual CTR B Value',
                   mode='lines',
                   line = dict(
                       color = ('rgb(205, 12, 24)'),
                       width = 2,
                       dash = 'dash'))

fig = go.Figure([trace0, trace1, trace2, trace3])
fig.update_layout(
    title='Beta Distributions for both Ads',
    xaxis={'title': 'Possible CTR values'},
    yaxis={'title': 'Probability Density'})


fig.show()
```
</p>
</details>


<iframe src="/mab-problem/3.html" width="700" height="450" frameborder="0"></iframe>


Note the intersection area. There might be the cases that value of Beta distribution for Ad A will be higher, than for Ad B, so algorithm will choose Ad A(which performs worse).


```python
fig = algorithm_performance(chosen_ads, total_reward, regret_list)
fig.show()
```


<iframe src="/mab-problem/4.html" width="700" height="450" frameborder="0"></iframe>


The regret is the lowest we've seen so far. Each uptick in regret happened when the Ad A was chosen. In the CTR Value plot, you can see that in the beginning, the green (Thompson sampled CTR value for Ad A) values were often greater than the tan (Thompson sampled CTR value for Ad B), resulting in Ad A being shown.

Note the differences in variations for each ad. The algorithm explores constantly, then naturally exploits the ad variant with the highest sample taken from the appropriate Beta distribution. This can be shown in the top plot of the distributions. The Beta distribution for Ad B is much higher and narrower than that of Ad A, meaning its sampled values will be consistently higher than those of Ad A, which is exactly what we want!


```python
thompson_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}
```

## Upper Confidence Bound (UCB1)

* *50% - Exploration*
* *50% - Exploitation*

Unlike the Thompson Sampling algorithm, the Upper Confidence Bound cares more about the uncertainty (high variation) of each variant. The more uncertain we are about one variant, the more important it is to explore. 

Algorithm chooses the variant with the highest upper confidence bound value (UCB) which represents the highest reward guess for the variant. It is defind as follows:

$UCB = \bar x_i + \sqrt{\frac{2 \cdot \log{t}}{n}}$ ,

where $\bar x_i$ - the (CTR rate) for $i$-th step,

$t$ - total number of (impressions) for all variants,

$n$ - total number of (impressions) for choosen variant

The logic is rather straightforward:

1. Calculate the UCB for all variants.
2. Choose the variant with the highest UCB.
3. Go to 1.


```python
regret = 0 
total_reward = 0
regret_list = [] 
ctr = {'A': [], 'B': []}
index_list = [] 
impressions = {'A': 0, 'B': 0}
clicks = {'A': 0, 'B': 0}

for i in range(n):
    
    win_ad = 'A'
    max_upper_bound = 0
    for k in ads:
        if (impressions[k] > 0):
            CTR = clicks[k] / impressions[k]
            delta = math.sqrt(2 * math.log(i+1) / impressions[k])
            upper_bound = CTR + delta
            ctr[k].append(CTR)
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            win_ad = k
    index_list.append(win_ad)
    impressions[win_ad] += 1
    reward = bernoulli.rvs(ACTUAL_CTR[win_ad])
    
    clicks[win_ad] += reward
    total_reward += reward
    
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
    regret_list.append(regret)
```


```python
fig = algorithm_performance(chosen_ads, total_reward, regret_list)
fig.show()
```


<iframe src="/mab-problem/5.html" width="700" height="450" frameborder="0"></iframe>


You can see that regret went up when the algorithm tried to decrease uncertainty of CTR for Ad A by choosing it.

It might be useful when you want the model to choose the best variant more often, but are still interested in reducing the uncertainty of both variants.


```python
ucb1_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}
```

## Comparison and Conclusions

Now let's compare four of this methods and see which one performed better for our problem.

First of all, it's obvious that we want to show the Ad B more often since its actual CTR is 0.65. Let's take a look at the ratio how many time the right Ad has been chosen for each algorithm.

<details><summary>Code</summary>
<p>

```python
trace0 = go.Bar(
    x=['Random Selection', 'Epsilon Greedy', 'Thompson Sampling', 'UCB1'],
    y=[random_dict['ads_count']['A'], 
       epsilon_dict['ads_count']['A'], 
       thompson_dict['ads_count']['A'],
       ucb1_dict['ads_count']['A']],
    name='Ad A',
    marker=dict(color='rgba(10, 108, 94, .7)'))

trace1 = go.Bar(
    x=['Random Selection', 'Epsilon Greedy', 'Thompson Sampling', 'UCB1'],
    y=[random_dict['ads_count']['B'], 
       epsilon_dict['ads_count']['B'], 
       thompson_dict['ads_count']['B'],
       ucb1_dict['ads_count']['B']],
    name='Ad B',
    marker=dict(color='rgba(187, 121, 24, .7)'))

fig = go.Figure([trace0, trace1])

fig.update_layout(
    title='Ratio of appearance of both Ads throughout the trials',
    xaxis={'title': 'Algorithm'},
    yaxis={'title': 'Ratio'},
    barmode='stack')

fig.show()
```
</p>
</details>


<iframe src="/mab-problem/6.html" width="700" height="450" frameborder="0"></iframe>


As you can see, three algorithms Epsilon Greedy, Thimpson Sampling and UCB1 showed Ad B most of the times (95%+).

<details><summary>Code</summary>
<p>

```python
trace0 = go.Scatter(
    x=np.arange (0, n, 1),
    y=random_dict['regret_list'],
    name='Random Selection',
    marker=dict(color='#ffcc66')
)
trace1 = go.Scatter(
    x=np.arange (0, n, 1),
    y=epsilon_dict['regret_list'],
    name='e-Greedy',
    marker=dict(color='#0099ff')
)
trace2 = go.Scatter(
    x=np.arange (0, n, 1),
    y=thompson_dict['regret_list'],
    name='Thompson Sampling',
    marker=dict(color='#ff3300')
)
trace3 = go.Scatter(
    x=np.arange (0, n, 1),
    y=ucb1_dict['regret_list'],
    name='UCB1',
    marker=dict(color='#33cc33')
)

fig = go.Figure([trace0, trace1, trace2, trace3])

fig.update_layout(
    title='Regret by the Algorithm',
    xaxis={'title': 'Trial'},
    yaxis={'title': 'Regret'}
)

fig.show()
```
</p>
</details>


<iframe src="/mab-problem/7.html" width="700" height="450" frameborder="0"></iframe>


Taking to account that Thompson Sampling and Epsilon-Greedy algorithms chose ad with the higher CTR (B) most of the time, it shouldn't come as surprise that their regret values are the lowest.

<details><summary>Code</summary>
<p>

```python
trace0 = go.Bar(
    x=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],
    y=['UCB1', 'Thompson Sampling', 'e-Greedy','Random Selection'],
    orientation = 'h',
    marker=dict(color=['#33cc33', '#ff3300', '#0099ff', '#ffcc66']),
    opacity=0.7
)

trace1 = go.Scatter(
    x=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],
    y=['UCB1', 'Thompson Sampling', 'e-Greedy', 'Random Selection'],
    mode='text',
    text=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],
    textposition='middle left',
    line = dict(
        color = ('rgba(255,141,41,0.6)'),
        width = 1
    ),
    textfont=dict(
        family='sans serif',
        size=16,
        color='#000000'
    )
)

fig = go.Figure([trace0, trace1])

fig.update_layout(
    title='Total Reward by Algorithms',
    xaxis={'title': 'Total Reward (Clicks)'},
    margin={'l':200},
    showlegend=False
)

fig.show()
```
</p>
</details>


<iframe src="/mab-problem/8.html" width="700" height="450" frameborder="0"></iframe>


It can be the case that total reward for the algorithm with the lowest regret value will not be the highest. It is caused by the fact that even if the algorithm chooses the right ad it doesn't guarantee that the user will click on it.

As was told from the beginning, the Thompson Sampling is generally the best choice, but we also looked at other algorithms and discussed how and when they might be useful. The method you choose depends on your unique problem, prior information you have and what information you want to receive afterwards. 
