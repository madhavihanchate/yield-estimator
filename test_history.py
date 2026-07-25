import urllib.request, json
r = urllib.request.urlopen('http://127.0.0.1:8000/api/history/?district=Bagalkot&crop=Brinjal', timeout=10)
d = json.loads(r.read())
years = d['years']
print(f"Years: {len(years)}")
for y in sorted(years.keys()):
    v = years[y]
    print(f"  {y}: yield={v['yield']} area={v['area']} production={v['production']}")
