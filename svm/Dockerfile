FROM continuumio/anaconda3
RUN mkdir /svm
WORKDIR /svm
COPY ./src/* /svm/
CMD ["python","./svm_anneal_uci.py"]
