Introduction
Background/Motivation
The project will focus on the sentiment, financial, market and economic analyses based on a five-year period from 2021 to 2026 of the following data: stock close prices, balance sheets, income statements, and financial new sentiments of the top five banks of Canada, the Target Overnight Rate of Canada and CPI Core Trim. A key objective is to address the financial health of each bank to determine the viable decision-making for investors. Therefore, the approach is to create an investment overview through deep learning models, such as LSTM, BERT and ChatGPT models, to mitigate against market volatility. To further emphasize, it is significant to add a layer of deeper analysis of the financial markets and its economic impact to current news sentiments, as it addresses the ambiguity associated with experienced versus novel investors in day trading. The objective is enlarging an investors’ lens by providing a wide aperture of insights in today’s financial emerging markets with a sharper focus on the risk appetite and economic implications associated with investment decision-making using advanced machine learning models and data analytical techniques, such as preprocessing, data wrangling, data validation, data visualization and so forth. 

Problem Statement
Investors are presented with challenges when analyzing the financial market in relation to the top five banks of Canada. Gap of the lack of financial acumen, market implications, and in relation to the economy, remains at large when investors make financial decisions in real-time. Scaling time series data presents its challenge when accounting for noise, seasonality, and missing data. TO sum up, inexperienced investors who are not adequately educated with financial acumen will fail to address the wider scope of the market and economic implications on their investment decisions.

Objectives/Contributions
The research engages with the integration of deep neural models, such as LSTM, TimeGPT, BERT and advanced algorithmic training to improve metrics, such as test MAE, total accuracy, precision, recall and F1 score for better investment decision making opportunities without compromising the uncertainty or the market’s external factors. 
The Primary contributions of the work include:
•	Combing a five-year financial sentiment to the stock close price for the top five banks in Canada and addressing the economic impact using the macroeconomic indicator, Core CPI Trim, and the Target Overnight Rate of Canada. 
•	Systematic evaluation of data science techniques, such as data preprocessing, feature engineering, model validation, and data visualization to provide a comprehensive overview of advanced deep learning techniques and its application to enhance the financial decision-making intelligence of investors
•	Using real-world research with a recent 5-year period from 2021 to 2026 to conduct an empirical comparison of machine learning classifiers, such as Bidirectional Encoder Representation Transformers (BERT) and Chat Generative Pre-trained Transformer (GPT) models, and in addition, metrics, such as precision (True Positive (TP)/ TP+ False Positive (FP)), recall (TP/ TP True Negative (TN)), F1 score (2 x Precision x Recall/Precision + Recall) and test Mae derived from financial, sentiment, market, and economic analyses
•	Highlight and analyze the challenges and limitations of the study and identify the scope for potential research 
•	A framework of recommendations to mitigate regulatory changes, competitive disruptions, and cash shortage pitfall 


Research Questions:

1)	After the deployment of the BERT and ChatGPT models, how can a comparison analysis of the sentiment scores and metrics, such as high accuracy, precision and recall be conducted for the top five banks of Canada?
2)	How can the financial posture and risk appetite be determined using liquidity ratios, such as current ratio, quick ratio and solvency; leverage ratio, such as debt-to-equity ratio; profitability ratio, such as return on equity; and market value ratio, such as price-to-earnings ratio, to assess the financial posture and risk appetite of each of the top 5 banks in Canada?
3)	Based on the risk assessment of market volatility impact on the financial market, how can the fair pricing of each stock be determined in conjunction with the pricing behaviour and trends of a 5-year forecast period of the top five bank stocks’ ‘Close Price’ via the implementation of the LSTM, TimeGPT and PSO models?
4)	How can the effectiveness of each of the top 5 banks ‘Close Price’ be used to allocate regulatory changes, such as competitive disruptions and cash shortage pitfall based on the macroeconomic indicator, such as a five year period from 2021 to 2026 of a seasonally adjusted Core Consumer Price Index Trim, and the Overnight Rate of Canada?


Methodology
Research Design
•	Classification techniques deep learning models, such as BERT, FINBERT and ChatGPT, were used to classify the financial sentiments into categories of positive (1) and negative (-1) and evaluation techniques included training time, total accuracy, precision, recall, and F1 score.
•	TimeGPT and LSTM used resampling of all the top five banks’ ‘Close Price’
•	Metric for LSTM is test Mae
•	Metrics for BERT and ChatGPT models are precision, recall and F1 Score
•	Python library: Tensorflow, keras – better understanding to compute in comparison totorch syntax
•	Training time was observed and recorded for all models 
•	Exploratory ata analysis will be implemented using matplotlib and seaborn python libraries to generate graphs for financial ratios, sentiments, test scores, and so forth
•	Preprocessing step includes filling NaNs with 0s based on the implementation of the lambda python library

LSTM
Output shape = 4, 64	Input shape – 64 x 32 x 32	Dropout rate = 0.5
Kernel regularizer = regularizer.12(0.001)	Batch size 
200	return_sequences=True

Metrics : MAE	Validation split 
0.2	optimizer=Adam

Loss
mean squared error	input_shape=(time_steps, features)
	learning_rate=0.0001

Epoch 
100	Verbose 1	return_sequences= True)


ChatGPT
Positive 
1	Negative 
-1	Neutral 
0	
			


BERT
Random_state = 42/43	Test_size= 0.2	Input(shape=(), dtype=tf.string, name='text')
	learning_rate=3e-5
	epochs = 60

name='BERT_encoder'
	trainable=False,
	    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
	f_keras.optimizers.RMSprop
	batch_size = 24



PSO -LSTM Optimization technique

  Learning Rate: 0.0100000
	Optimizer
Adam	Output dim
100	Learning rate
le-5	input_length
max_seq_len
Metric
accuracy
	Epoch 
10	Batch size

32	Verbose 
0	input_dim

validation_data
(X_val, y_val),
	loss
'sparse_categorical_crossentropy'
	max_seq_len = 100	POP_SIZE= 10
	tokenizer_vocab_size

num_words=10000,	test_size=0.2
	padding='post'
	truncating='post'
	random_state=42

w= 0.2
	c1 = 1
	c2 = 2
	  LSTM Layer 1 Units: 108
  LSTM Layer 2 Units: 33
	  Dropout Rate: 0.3536

Test sequences shape: (2, 95)
	Validation sequences shape: (8, 95)
	Train sequences shape: (40, 95)
	Max sequence length: 95
	Tokenizer vocab size: 780


In this proposed study of the financial stock market, multiple time series datasets were used. For example, “Canada_top_5_bank_news (3).csv”, “stock_data_2021_06_19_to_2026_06_19.csv”, “top_five_bank_balance_sheets 2.csv”, “top_five_bank_income_statements 2.csv”, “target_overnight_rate_canada.csv”. The first time series dataset, “stock_data_2021_06_19_to_2026_06_19.csv” includes 1247 rows and 25 columns. The stock market dataset is gathered from a period between 2021-06-19 to 2026-06-19 and it has names, such as “price” and “ticker”. The features of the dataset include the following column names:  ‘Close: BMO’, ‘Close: BNS’, ‘Close: CM’, ‘Close: RBC’,’ Close: TD’, ‘High: BMO’, ‘High: BNS’, ‘High: CM’, ‘High: BNS’, ‘High: CM’, ‘High: RBC’,’ High: TD’, ‘Low: BMO’, ‘Low: BNS’, ‘Low: CM’, ‘Low: RBC’, ‘Low: TD’, ‘Open: BMO’, ‘Open: BNS’, ‘Open: RBC’, ‘Open: TD’, ‘Volume: BMO’, ‘Volume: BNS’, ‘Volume: CM’, ‘Volume: RBC’, and ‘Volume: TD’. The data type of the stock market indicators, Close, High and Low, across the top five banks of Canada, TD, BMO, RBC, CIBC and Scotiabank, is float64 and for the indicator, Volume, is inter64. The first dataset was obtained from YFinance python library and extracted to excel Comma Separated file (CSV). 
The second dataset, “top_five_bank_income_statements 2.csv” had a shape 84 rows and 30 columns. The Income Statement for all of the banks were indexed by metrics in tuples, such as ‘Tax Effect Of Unusual Items’, ‘Tax Rate For Calcs’, ‘Normalized EBITDA’, ‘Total Unusual Items’, ‘Total Unusual Items Excluding Goodwil’, ‘Net Income From Continuing Operation’, ‘Net Minority Interest’, ‘EBITDA’, ‘EBIT’, ‘Net Interest Income’, ‘Interest Expense’, ‘Normalized Income’, ‘Net Income From Continuing And Discontinued Operation’, ‘Total Expenses’, ‘Total Operating Income As Reported’, ‘Diluted Average Shares’, ‘Basic Average Shares’, ‘Diluted EPS’, ‘Basic EPS’, ‘Diluted NI Avail to Com’, ‘Net Income Common Stockholders’, ‘Preferred Stock Dividends’, ‘Net Income’, ‘Net Income Including Noncontrolling Interests’, ‘Net Income Continuous Operations’, ‘Tax Provision’, ‘Pretax Income’, ‘Other Income Expense’, ‘Other Non Operating Income Expenses’, ‘Special Income Charges’, ‘Gain On Sale Of Ppe’, ‘Restructuring And Mergern Acquisition’, ‘Net Non Operating Interest Income Expense’, ‘Total Other Finance Cost’, ‘Interest Expense Non Operating’, ‘Operating Income’, ‘Operating Expense’, ‘Provision For Doubtful Accounts’, ‘Depreciation Amortization Depletion Income Statement’, ‘Depreciation And Amortization In Income Statement’, ‘Amortization’, ‘Amortization Of Intangibles Income Statement’, ‘Selling General And Administration’, ‘Gross Profit’, ‘Total Revenue’ and ‘Operating Revenue’. Data type is object and the data ranges from 2026-03-31 to 20221-10-31. The second dataset was obtained from Yfinance python library and extracted to excel CSV form. 
The third dataset, “top_five_bank_balance_sheets 2.csv” ranges from 2026-03-31 to 20221-10-31 and it has 50 rows with the following column names: ‘Non Current Deferred Liabilities’, ‘Non Current Deferred Liabilities’, ‘Non Current Deferred Taxes Liabilities’, ‘Long Term Debt And Capital Lease Obligation’, ‘Long Term Capital Lease Obligation’, ‘Long Term Debt’, ‘Current Liabilities’, ‘Other Current Liabilities’, ‘Current Deferred Liabilities’, ‘Current Deferred Revenue’, ‘Current Debt And Capital Lease Obligation’, ‘Current Capital Lease Obligation’, ‘Current Debt’, ‘Other Current Borrowings’, ‘Pension and Other Post Retirement Benefit Plans Current’, ‘Payables And Accrued Expenses’, ‘Current Accrued Expenses’, ‘Interest Payable’, ‘Payables’, and ‘Dividends Payable’. The third dataset was obtained from Yfinance python library and extracted to excel CSV form.
The fourth dataset, “target_overnight_rate_canada.csv", was obtained from the Bank of Canada’s website and extracted as CSV format. It has 1306 rows and 1 column. Data is indexed and days of week filtered for bank holidays and weekends. The column name names are ‘Date’ and ‘Target Overnight Rate of Canada’. The fifth dataset is “stock_data_2021_06_19_to_2026_06_19.csv” was extracted from Yahoo Finance python library and saved to a CSV file using python code from the Google Collaboratory Notebook. It has 1249 rows and 26 columns and a data type of an object. The names of the columns included the following: 'Price', 'Close', 'Close.1', 'Close.2', 'Close.3', 'Close.4', 'High’, 'High.1', 'High.2', 'High.3', 'High.4', 'Low', 'Low.1', 'Low.2', 'Low.3', 'Low.4', 'Open', 'Open.1', 'Open.2', 'Open.3', 'Open.4','Volume', 'Volume.1', 'Volume.2', 'Volume.3', and 'Volume.4'. Note, the stock dataset ranges from 2021-06-19 to 2026-06-19. 
The fifth dataset, “1810025602-noSymbol.csv”, seasonally monthly adjusted. was extracted from Statistics Canada and ranges from a period of 2021-06-01 and 2026-06-01.it contains 61 rows and 1 column. The column names are Date, which is indexed and CPI Core Trim. The data type Is float64. Lastly, the six dataset is “top_five_bank_income_statements 2.csv” has columns, “id” and “content” for the top five banks of Canada . The data type is object. The columns were renamed to “title”, “summary” and “Ticker_Mentioned” and then separated into news pertaining to each bank, such as TD, RBC, BMO, Scotiabank, and CIBC. The data shape is 50 columns by 8 rows. The sentiment data was extracted from the defeatbeta api python library and then, saved to excel on Google Collarboratory.
The approach was to conduct financial and sentiment analyses through the implementation of deep learning techniques, such as LSTM, BERT, and ChatGPT. The sentiment data after encoding was structured as binary. all the datasets, The preprocessing techniques included removing all the null values using Lambda python library, renaming the data frames and making copies without compromising data integrity. Furthermore, data analytics techniques include data collection, data cleaning, data validation, data wrangling, descriptive analytics, resampling for TimeGPT, data visualization and analyzing statistical calculations for driving insights and identifying patterns. Both the Income Statement and Balance Sheet datasets were indexed by metrics and then, a simplified index was applied to all the data frames of the top five banks of Canada. The ratios which were calculated were current ratio, quick ratio, return on equity, debt to equity, debt to asset, asset turnover, and price per earning ratio.  

Anticipated Outcome:

BERT will outperform ChatGPT in sentiment analysis of the top 5 banks in Canada because it is bidirectional, higher precision on classification and sentiment analysis and extracting information with high relevancy to existing information. 
RBC will outperform other banks based its liquidity, profitability, efficiency and solvency ratios.
The market will experience volatility due to high inflation, stock prices will be low.
