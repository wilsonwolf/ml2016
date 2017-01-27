library(httr)
library(jsonlite)
options(stringsAsFactors = FALSE)

###Pull Summary Stats

dataTest <- GET(url = "https://api.blockchain.info", path = "stats")
dataTest$content <- rawToChar(dataTest$content) ##Convert to text from Unicode
stats <- fromJSON(dataTest$content) ##Parse JSON to List

###Pull Chart Data

chartTest <- GET(url = "https://api.blockchain.info", path = "charts/transactions-per-second?timespan=5weeks&rollingAverage=8hours&format=json")
chartTest$content <- rawToChar(chartTest$content)
chart <- fromJSON(chartTest$content)
chart <- chart$values

##Pull Exchange Data

delayPrice <- GET(url = "https://blockchain.info", path = "ticker")
delayPrice$content <- rawToChar(delayPrice$content)
exchangeRate <- fromJSON(delayPrice$content)
exchangeRateDollar <- exchangeRate$USD

### Bitfinex API
