FROM python:3.6

ADD requirments.txt /app/
WORKDIR /app
RUN pip install -r requirments.txt

ADD . /app

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:app.server"]