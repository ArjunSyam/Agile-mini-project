FROM --platform=linux/x86_64 python:3.9

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python","app.py"]
