# Moving Edge Cloud route finder
## 1. Docker Swarm mode
###  - install the docker on raspberry pi
###  - docker swarm init --advertise-addr (ip address) in Manager node  
###  - docker swarm join \--token in Worker node  
###  - Check setting as 'docker node ls' 
 
 
## 2. communications between raspberry pi
###  - manager node, worker node assign a static IP 
###  - manager node -> using eth1 (usb port),  worker node -> using eth0 (Lan port) 
###  - Set worker node gateway to manager node IP 
###  - Connect two usb to lan cables to the master node and connect to the worker lanport with a cross cable 


## 3. Settings for Remote Development 
###  - install the Pycharm on windows
###  - write python code 
###  - Remote debug to raspi using Ssh tunneling for using in raspberry pi’s environments 
###  - push and Merge to remote store 
