import requests
import json
import os 
import sys


def send_message(message):
    webhook = os.environ.get("webhook_slack")
    data = {"text":message}
    requests.post(webhook,json.dumps(data))


if __name__=='__main__':
    send_message(f"Done training {sys.argv[1:]}")
