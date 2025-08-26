# Stock-Market-Prediction
Using Alpaca API I am trying to simulate how much money one could make using $20,000 in the stock market. <br>
I have chosen to use a LSTM(long short term memory model) to predict whether to purchase or sell the stock. <br>



While predicting stock prices is a gamble since there are so many different factors influencing the price other than previous days, there is still a good oportunity to make money in the short term. This is done using a LSTM model, this model is designed to take into account sequential data and long term dependencies. Here is a great video explaining what an LSTM is (https://www.youtube.com/watch?v=YCzL96nL7j0)
<br> <br>

## Goals
For this project I want to be able to show that using machine learning predicting the stock market and making a profit is possible. Many people have any extra money they have in a savings account that might not even make 1% apy. By using machine learning we can prove that it is possible to make more money than the average savings account.

## Steps
To get started with this project we will need to get some data from the stock market, I was recommended using Alpaca API (https://alpaca.markets/) and it turns out it works very well for stock market data. First we will need an idea of what stocks to invest in. 
I have 8 different companies that I am going to be investing in. 
<br> <br>
JPM: JP Morgan: financial service <br>
KULR: KULR Technology Group Inc: technology company <br>
META: Meta: technology company <br>
MS: Morgan Stanley: financial service <br> 
MU: Micron: technology company, produces computer memory and data storage <br>
NVDA: Nvidia: computer manufacturing company <br>
OKLO: OOKLA: digital media and internet company <br>
AVGO: Broadcom: technology company, creates semiconductors <br> <br>
I have chosen these 8 different companies to work with because of the potential growth in each of them. There are so many different companies to invest in that these 8 will work for now, there is potential to add more or take some out. 

Now that we have the stocks and the data the next step is to clean and scale the data to make it usable. Scaling the data is needed in a model like LSTM due to being trained with gradient descent. Once the data is cleaned and scaled we can now train the model. 
Using the trained model we can then predict and determine if we should purchase the stock. This is the tricky part on trying to make a profit. Determining when to purchase and sell a stock is a very hard thing to do day by day because of how volitile the market can be. I have set a threshold of 1% to determine if it should purchase or sell the stock. In one day the market will not change very much which is why I chose 1% difference. The S&P 500 changes about 1% in a day on average so anything higher might be difficult to make a profit off of. 

## Outcome
As of writing this this will need to be updated and be fixed to show how well this model does. 

