FROM python:slim

RUN pip install uv

WORKDIR /app
COPY . .
RUN uv pip install --no-cache --system -r requirements.lock

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EXPOSE 8080
