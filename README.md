how to run this app.

1). Clone the repo
2). move to folder BRAIN TUMOR DETECTION
3). create a virtual env using virtualenv 'environment name'.
4). Now type . \Scripts\activate to activate your virtualenv venv
5). install packages required by the app by running the command pip install -r requirements.txt if you found error of torch or torch-vision libraries so you candownload it from below commands. for windows cpu torch library command with conda: conda install pytorch torchvision cpuonly -c pytorch for windows cpu torch library commad with pip: pip install torch==1.6.0+cpu
torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

For Linux,Mac,CUDA version for windows and much more you can visit https://pytorch.org

1). now run the app using flask run.
2). if all things work fine then you can view running web application in your browser at port 5000.

Download the model file from 
https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view?usp=sharing and add to model directory in order to run  the project.
