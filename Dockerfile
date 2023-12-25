FROM python:3.11-slim

LABEL maintainer="hasanain@aicaliber.com"

EXPOSE 80

WORKDIR /model_serving

COPY /model_serving/requirements.txt /model_serving/setup.py /model_serving/README.md ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./model_serving/ .


CMD ["python", "src/main.py"]