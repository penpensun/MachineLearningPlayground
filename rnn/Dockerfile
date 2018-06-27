FROM continuumio/anaconda3
RUN pip install --upgrade pip && \
  pip install numpy && \
  pip install tensorflow 
RUN mkdir /workspace/
WORKDIR /workspace/
COPY ./src_rnn1/* ./
CMD ["python","rnn_poetry.py"]


