FROM zironycho/pytorch:1120-cpu-py38

LABEL maintainer="Salil Gautam <salil.gtm@gmail.com>"
LABEL description="Dockerfile for Assignment 9 of EMLOv3 - AWS."

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY demo/ demo/
COPY configs/ configs/

RUN pip install --no-cache-dir -r requirements.txt
RUN rm -rf /root/.cache/pip

EXPOSE 80

CMD ["python3", "demo/gpt_gradio.py"]