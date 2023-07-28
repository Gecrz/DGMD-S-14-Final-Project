#download the data using the below code into your local setup
url = 'http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz'
file_path = 'BreaKHis_v1.tar.gz'

response = requests.get(url, stream=True)

if response.status_code == 200:


  total_size = int(response.headers.get('content-length', 0))
  progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

  with open(file_path, 'wb') as file:
      for chunk in response.iter_content(chunk_size=1024):
          if chunk:
              file.write(chunk)
              progress_bar.update(len(chunk))

  progress_bar.close()
  print('File downloaded successfully.')
else:

  print('Failed to download the file.')
