# Running this project on Google Colab

If you don't have Python set up locally, you can run the whole project on
Google Colab. Upload the project folder (or clone it into Colab), then run
each of the following in separate cells.

### 1. Install dependencies
```python
!pip install -r requirements.txt pyngrok -q
```

### 2. Generate the dataset
```python
!python generate_dataset.py
```

### 3. Train and compare models
```python
!python train_model.py
```

### 4. Set your ngrok authtoken
Get a free token from https://dashboard.ngrok.com/tunnels/authtokens
(never share or commit this token publicly):
```python
!ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 5. Launch the app and get a public link
```python
import os, time
from pyngrok import ngrok

ngrok.kill()
os.system("streamlit run app.py &")
time.sleep(6)
public_url = ngrok.connect(8501)
print("App is live at:", public_url)
```

Click the printed link, then click **"Visit Site"** on the ngrok warning
page (this is normal for free ngrok tunnels — it's not an error).

### 6. When you're done
```python
ngrok.kill()
```
