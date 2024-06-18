FROM python:3.6

COPY requirements.txt /
RUN python3 -m pip install --upgrade pip && \
pip3 --default-timeout=1000 install -r /requirements.txt --upgrade --ignore-installed -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ecci_sdk.py /
COPY edge_send.py /

WORKDIR /

CMD ["python3", "edge_send.py"]
