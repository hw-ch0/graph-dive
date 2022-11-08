FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN apt-get install -y git zip

RUN pip install jupyter sklearn numpy scipy ipython pandas torchsummary tqdm

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-geometric

RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

CMD ["/bin/bash"]
