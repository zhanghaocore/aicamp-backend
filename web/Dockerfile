FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app/
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY ./app/ ./
ENV CUDA_VISIBLE_DEVICES 1,2,3
ENTRYPOINT ["python", "serve.py", "--host", "0.0.0.0", "--port", "8080"]
