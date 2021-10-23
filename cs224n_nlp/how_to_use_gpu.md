# make GCP pytorch instance and transfer target folder to instance easily

https://github.com/cs231n/gcloud

1. At local
~~~ sh
$ ssh-keygen -t rsa -f ~/.ssh/"KEY_NAME" -C "USER_ID@MAIL_ADDRESS.com" 
$ cat ~/.ssh/"KEY_NAME".pub
$ sudo chmod 400 ~/.ssh/"KEY_NAME" 
$ sudo chmod 400 ~/.ssh/"KEY_NAME".pub   
$ sudo chmod 400 ~/.ssh/known_hosts
~~~

2. At gcp console 

* [make pytorch instance](https://console.cloud.google.com/marketplace/config/click-to-deploy-images/tensorflow)

* gcp console > Compute Engine > VM instances > 'instance_name' > edit
    - check "Allow HTTP traffic"
    - check "Allow HTTPS traffic".
    - add "cat ~/.ssh/"KEY_NAME".pub" key at "SSH Keys"

3. At gcp instance

~~~ sh
$ sudo passwd root
$ sudo su
$ sudo vi /etc/ssh/sshd_config
- PasswordAuthentication no -> yes
- PermitRootLogin no -> yes
~~~

4. At local

``` sh
$ ssh -i ~/.ssh/"KEY_NAME" "USER_ID"@"EXTERNAL_IP"
$ scp -i ~/.ssh/"KEY_NAME" -r "ORIGIN_FOLDER" "USER_ID"@"EXTERNAL_IP":/home/"USER_ID"/"DEST_FOLDER"
$ scp -i ~/.ssh/"KEY_NAME" "ORIGIN_FOLDER/TARGET_FILE" "USER_ID"@"EXTERNAL_IP":/home/"USER_ID"/"DEST_FOLDER"
$ scp -i ~/.ssh/"KEY_NAME" -r "USER_ID"@"EXTERNAL_IP":/home/"USER_ID"/"GCP_DEST_FOLDER" "LOCAL_ORIGIN_FOLDER" 
```

app. tips

> use 'screen' to keep learning: screen -S 'screen_name'

> to detach screen: Ctrl + A + D


# run py files on GPU at Google Colab

1. Select to use GPU
> Colab > Runtime > Change Runtime Type > Hardware Accelerator > GPU

2. On Colab Notebook
~~~ python

# check whether GPU can be deployed
import torch
torch.cuda.is_available
torch.cuda.device_count()
torch.cuda.get_device_name(0)

# check hardware information
!head /proc/cpuinfo
!head -n 3 /proc/meminfo
!nvidia-smi
%load_ext autoreload
%autoreload 2

# connect Google Drive and target folders
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

FOLDER_NAME = 'cs224n/a4'
assert FOLDER_NAME is not None, "[!] Enter the folder name."

import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDER_NAME))

# change directory
%cd /content/drive/My\ Drive/$FOLDER_NAME
!ls

# run bash file or py file with parameters
!sh run.sh train
!CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --batch-size=64 --cuda
~~~
