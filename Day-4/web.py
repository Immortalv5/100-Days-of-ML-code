import requests
from bs4 import BeautifulSoup

URL = 'https://coronavirus.data.gov.uk/#local-authorities'
page = requests.get(URL)

print(page.content)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id = "root")
print(results)
results = soup.find(id = "sc-fznxsB cUWXFh govuk-width-container")
print(results)
results = soup.find(id = "sc-fzpjYC gJohPa")
print(results)
