# 301-Project 1 
## Group 15: Aryan Nesti
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

