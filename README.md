# 301-Project 1 
## Milestone 1

###### Steps for Collab
1. I create a new repo under one Github account of a member of our team.
2. I have decided to use collab which is located https://colab.research.google.com/drive/1qLwhm5FAinETMUOtTnDyaDX2m8jD-q4F?authuser=1#scrollTo=tbaN8k2Dw9bv. Where I install NNI usung these lines of code:
```
! pip install nni # install nni
! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip # download ngrok and unzip it
! unzip ngrok-stable-linux-amd64.zip
! mkdir -p nni_repo
! git clone https://github.com/microsoft/nni.git nni_repo/nni # clone NNI's offical repo to get examples
```
3. Now we must create a ngrok account. After creating the account you verify your email and then click the ***Your Authtoken*** shown on the left underneath ***Setup & Installation***
4. You copy your authtoken and the paste it with this code
```
! ./ngrok authtoken YOUR_AUTH_TOKEN
! nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 &
get_ipython().system_raw('./ngrok http 5000 &')
! curl -s http://localhost:4040/api/tunnels # don't change the port number 4040
```
5. After running all the code you will see a url like http://xxxx.ngrok.io, after the last line of code and this will show the NNI's Web UI

###### Steps for Poetry
1. I create a new repo under one Github account of a member of our team.
2. I install[https://python-poetry.org/docs/#installing-with-the-official-installer] and setup a new poetry porject with `poetry innit -n`
3. Next we do `poetry shell` to be inside of the enviorment
4. Now we install poetry and setup the enviorment and use `poetry add` to install the dependencies 
5. Here I add NNI using the method in step 4 and then run `nni hello` this will download the `nni_hello_hpo` and send a command message stating `Please run "python nni_hello_hpo/main.py" to try it out.`
6. Finally you run that line and you will be prompted to two local links and those will be your NNI's Web UI

## Milestone 2 
For this project we are using Unet which is a convolutional neural network meaning it is applied to visual images. In this case we will apply Unet to satelite imagery that were taken of several different locations ranging from cities to deserts and oceans. On the left is the image taken and the image on the right is the effect of Unet. 
`More can be viewed in docs/baseline-performance.md`
<!-- Image here -->

![Figure_1](https://user-images.githubusercontent.com/98928740/200205279-83f298a4-5592-41a7-91c8-15774bfcbc52.png)
![Figure_2](https://user-images.githubusercontent.com/98928740/200205260-f1abc72a-ac6a-4091-a582-97a38e67fd38.png)

## Milestone 3
To further perfect our model we conducted Hyperparameter Optimization using NNI method Metis. We will work with hyperparameters like weight, leanring rate and droprate. For teh results please visit the `docs` folder down to ` hyperparameter-optimization.md` file.
```
search_space = {
    'weight1': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight2': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight3': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight4': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight5': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight6': {'_type': 'uniform', '_value': [0, 0.2]},
    'dropout_rate': {'_type': 'uniform', '_value': [0, 1]},
    'learning_rate': {'_type': 'uniform', '_value': [0, 0.1]},
}
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Metis'
experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize'
}
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
```