FROM python:3.13
RUN /usr/local/bin/python -m pip install --upgrade pip
WORKDIR /app
COPY app/ /app/
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["app/wildfire_app.py"]