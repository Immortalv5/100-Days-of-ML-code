import requests
from bs4 import BeautifulSoup

URL = 'https://www.monster.com/jobs/search/?q=Software-Developer'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id = "ResultsContainer")
job_elems = results.find_all('section', class_='card-content')
i = 1

for job_elem in job_elems:
    title_elem = job_elem.find('h2', class_='title')
    company_elem = job_elem.find('div', class_='company')
    location_elem = job_elem.find('div', class_='location')
    if None in (title_elem, company_elem, location_elem):
        continue
    print(i)
    i=i+1
    print(title_elem.text.strip())
    print(company_elem.text.strip())
    print(location_elem.text.strip())
print('\n\n')

python_jobs = results.find_all('h2', string=lambda text: 'microsoft' in text.lower())
print(len(python_jobs))

for p_job in python_jobs:
    link = p_job.find('a')['href']
    print(p_job.text.strip())
    print(f"Apply here: {link}\n")
