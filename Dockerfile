FROM python:3.13

WORKDIR /usr/src/app.

COPY  ./Utilities ./

RUN pip install --no-cache-dir -r ./requirements.txt

EXPOSE 5000

CMD ["python", "server.py"]