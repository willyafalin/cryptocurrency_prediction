# Portfolio Optimisation of Cryptocurrencies using Machine Learning

[![Onyxia](https://img.shields.io/badge/Launch-Datalab-orange?logo=python)](https://datalab.sspcloud.fr/launcher/ide/vscode-python?name=vscode-python&version=2.1.19&s3=region-ec97c721)


Welcome to the repository of our [Year 2 Ensae Data Science project](https://pythonds.linogaliana.fr/) !  

# 1. Presentation of the project 

The topic of our study is to **build an optimised portfolio of cryptocurrencies**, by maximising the expected gains over a given period of time and arbitraging this against a given risk profile of an investor. 

To do so, we followed three steps:  

(1) **Price prediction (the expected gains)**. We first determined a suitable model to predict prices (our expected gains),

(2) **Volatility forecasting (Risk profile)**. Another model is then identified to predict future volatility (our given risk profile).

In turn, these two aforementioned models allow us to:

(3) **Build an efficient frontier** that models this risk-return arbitrage in finance theory. 

In short, optimising an efficient frontier allows to get the required proportions to invest in certain cryptocurrencies as we model this risk-return arbitrage, which is the goal of our study. 

# 2. Motivation 

The motivation behind our study stems from both the unique characteristics of cryptocurrency as a financial asset and the rampant transformative potential of machine learning in modern society.

**Taming the potential of cryptocurrencies** (more particularly Bitcoin) **as a possible 'safe haven' asset** ('valeur refuge'). Understandably, Bitcoin is becoming seen more and more as a 'safe haven' asset. Indeed, its compounded returns are way more attractive than traditional financial instruments, such as bonds ('obligations'), insurance products or savings accounts. Big warning however, is its extreme volatility: Bitcoin is not only seen as a potential future 'safe haven', but also as a financial bubble ready to burst anytime. Therefore, successfully balancing this risk-return trade-off through portfolio optimisation could pave the way for using Bitcoin and other cryptocurrencies as a viable safe-haven asset in the future (though mostly unlikely at the present moment. As we bear this in mind, our project is a first attempt in achieving that aim). 

**Additional reason to tame this risk-return potential of crypto: the current growth of crypto market value**. Recent news such as the re-election of Donald Trump caused a steady increase of the value of Bitcoin. On top of this, current global geopolitical and the economic disruptions that followed are one of the many drivers for finding new safe assets to invest in. 

**The use of Machine learning in our project**. On a more personal note, it is increasingly established that Machine Learning has a transformative potential in this century, with far-reaching societal implications. Our project reflects our interest in applying machine learning models to address real-world problems, more specifically applied to financial economics issues. By leveraging ML models that are groundbreaking, we can apply this innovation to the economy. Here, targeting a specific segment of the financial market. Over time, such approaches could even improve the management of financial markets on a broader scale.

In this project, we try to address these three motivations.

# 3. Data

The data we retrived in this project come from APIs. In our exploration phase of the project, we used Kraken API, Twelvedata API and Binance. We opted for the Binance API, as it gives us more covariates to use for our predictions. 

In the first part of our project, we only used data on the Bitcoin cryptocurrency to determine the models we would use to build the efficient frontier, as presented in 1.1 and 1.2 (Prediction of crypto price at the close of the day, and prediction of the volatility of the asset).

One last thing to note on the data, is that we used daily intervals first for determining a suitable model. After that the predictions of a model was considered comparably better than other models, we used hourly data, in order to have more data points for a given period of time. This allowed to use less complex data as well as reducing the time to run codes. 

# 4. Plan of the project (Structure of the Main file)

Throughout the sections II. and III. where we modelise gains and volatility, we provide descriptive statistics over our dataset to support the modelisation as well.

- I. Data Extraction and Cleaning:

    - I.1 Scraping data using the Binance API, and data cleaning
    - I.2 Enriching the dataset with additional variables for models requiring several features (II.2, II.3, II.4)
    - I.3 Preparing data for ARMA modeling (II.1)

- II. Modelisation of expected returns :
II.1 to II.4 are models we don't retain, but support the choice for the use of Machine learning for such data. Therefore, we will retain the LSTM model (II.5)

    - II.1 ARMA modelling
    - II.2 Simple Linear Regression
    - II.3 Multiple Linear Regression
    - II.4. Support Vector Regression
    - II.5. Long short-term memory Model (LSTM)

- III. Modelisation of volatility : the GARCH model

- IV. Dashboard : more data collection and cleaning

After having stabilised the models for gains (II.) and volatility (III.) by using one set of cryptocurrency, we can collect more data for the subsequent analysis of the efficient frontier. This is what we do with our dashboard. 

The dashboard automatises the data collection over a portfolio of 10 cryptocurries. There is an option to select a given timeframe and a specific crypto to plot for. Then, the plots of the volatility and expected returns for that selected crypto and period are shown.

- V. Closing the model : building the efficient frontier
Using the returns and volatlities from the same portfolio of 10 cryptos in the dashboard, we present our final result.

# 5. Getting started, for replication of the project  

First, run the dependencies with the terminal. The dependencies are stored in `requirements.txt`. Then, run the `function2.py` code that stores all the functions needed to run the main do notebook `main_project.ipynb`.

Then, the main file `main_project.ipynb` allows to run the sections I., II., III.  

The section IV. (Dashboard) is run with the `dashboard.py` code. First, run the code `dashboard.py`. Then in the Terminal, type the command $\texttt{streamlit run dashboard.py}$ . You will be given a local host website link (the first link), which allows to access the dashboard. For this section, we advise using an IDE (such as visual code studio) and launch the aforementioned snippet in the terminal (click [here](https://datalab.sspcloud.fr/launcher/ide/vscode-python?name=vscode-python&version=2.1.19&s3=region-ec97c721) to use SSP cloud with vs-studio, or on the orange badge on top of this page).

So the full code in the terminal to run in order to reproduce the entirety project is such:

    ```
    pip install -r requirements.txt
    python3 function2.py # from here, the main file can be run entirely
    
    python3 dashboard.py
    streamlit run dashboard.py # then, select the localhost website link
    
    ```




