# cryptocoin-tensorflow-demo

by Joe Hahn<br />
jmh.datasciences@gmail.com<br />
21 January 2018<br />
git branch=gpu-on-bitfusion

### Summary:

This demo trains an LSTM neural network to predict daily changes in the
Ethereum cryptocurrency. Source code is a Jupyter notebook that uses Keras on Tensorflow
to build a simple LSTM neural network to predict daily changes in Ethereum. This model is
trained on a very narrow dataset, namely the daily values and volumes of Bitcoin and Ethereum.
The following implicitly assumes that Bitcoin movements are driving the Ethereum valuations,
which is at best partly true and certainly not sufficient for building an adequate predictive model
for Ethereum. But the principal goal here is to build and test an LSTM model using
a simple dataset, and that at least is achieved.

### Setup:

2 Launch a g2.2xl EC2 instance in AWS via recipe detailed in  
https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac
using these settings:

    EC2 > launch instance > Community AMIs
    search for 'Bitfusion Ubuntu TensorFlow' > g2.2xlarge ($0.76/hr)
    set tag Name=tf-demo
    security group settings:
        set SSH and TCP entries to have Source=My IP (this enables ssh and jupyter)
        add custom TCP rule, port=6006, Source=My IP (to enable tensorboard)
    create & download keypair named tf-demo.pem
    Launch

3 store tf-demo.pem in in subfolder named 'private' and set its permissions:

    chmod 400 private/tf-demo.pem

4 obtain the instance's public IP address from the EC2 console, and then ssh into the instance:

    ssh -i private/tf-demo.pem ubuntu@ec2-54-245-199-248.us-west-2.compute.amazonaws.com


5 clone this repo and select desired branch:

    git clone https://github.com/joehahn/cryptocoin-tensorflow-demo.git
    cd cryptocoin-tensorflow-demo
    git checkout gpu-on-bitfusion

6 install additional python libraries

    sudo pip install seaborn
    sudo pip install lxml
    sudo pip install --upgrade pandas            #install --upgrade to resolve version conflict
    sudo pip install --upgrade BeautifulSoup4

7 update locate database:

    sudo updatedb

8 get instance ID:

    ec2metadata --instance-id

10 start jupyter:

    jupyter notebook

11 browse jupyter at public_IP:8888 and log in with password=instance-id

    ec2-54-245-199-248.us-west-2.compute.amazonaws.com:8888

12 use Jupyter UI to upload predict_crypto_price.ipynb from desktop, and run it

13 monitor GPU usage via:

    watch -n0.1 nvidia-smi

14 Note that the LSTM model used here was cribbed from David Sheehan's blog post
https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning,
which is worth a read.

### Execute

The notebook downloads two years of bitcoin and ethereum prices:
![](figs/ethereum.png)
and plots currency prices
![](figs/price.png)
and volumes versus time:
![](figs/volume.png)
An LSTM (Long Short Term Memory) model will be trained on data 
accrued prior to 2017-11-15 (blue curve, below)
and that model will then be used to predict the next-day change in ethereum's price
during subsequent days (green curve)
![](figs/training.png)
To help the model predict ethereum's next-day price change, the model is trained
on 4 days of lagged price and volume data. The notebook then builds a simple
LSTM  neural network using Keras on top of Tensorflow;
this network has 2 hidden layers that are all 12 neurons wide,
and training requires about 1 minute using a Mac laptop's CPU. 
![](figs/lstm.png)
The MAE (mean absolute error) loss function is used to train the LSTM model,
and the model's MAE versus training epoch is shown below
![](figs/loss.png)
This model is trained to predict ethereum's _fractional_ next-day price, so this figure
tells us that the trained LSTM model can predict
ethereum's next-day price with a 5% accuracy.

The trained LSTM model is then applied to
the test dataset, to predict ethereum's next-day fractional price
change for all dates after 2017-11-15. Green curve (below)
shows the actual next-day price variation versus date,
while the blue curve shows the predicted price change. Although the model predictions are
in the desired neighborhood, those predictions do not recover ethereum's
actual next-day price variation with enough accuracy to want to invest.
Also this model's predictions on the test dataset also have MAE = 5% (same as earlier),
so there appears to be no sign of under/over fitting.
![](figs/prediction.png)
Lastly, the red curve in the above plot shows predictions made by a simple linear regression (LR)
that was also trained on this data; that curve shows that the LR model is only somewhat useful across
the first month of testing data, with the LR model then veering away from reality at later times.
The LR model's MAE was also twice that of the LSTM model, so LSTM was two times more accurate than the
simplest of all machine-learning algorithms, and the LSTM predictions were much better behaved
further into the future.

### Conclusions

The above illustrates how to fit a simple LSTM neural network, using Keras on top of Tensorflow
plus a modest amount of cryptocurrency data, executing inside a Jupyter notebook, see 
https://github.com/joehahn/cryptocoin-tensorflow-demo/blob/master/predict_crypto_price.ipynb
for additional details

### Next steps:

2 broaden the dataset used here to include other market data (easy) plus 
market & cryptocurrency news (challenging)

###Notes

1 execute python2 script at commandline:

    python2 ./predict_crypto_price.py

