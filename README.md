# Volume Calculation

In this project, you can use your pre-trained model and Mask-RCNN to calculate the volume of the object you have defined in real dimensions, obtaining values closest to the actual size.

# Jupyter Notebook

You can run the project in a notebook and observe the results by following the steps in the predict.ipynb file.

# Docker API Application

Additionally, you can dockerize this application. The Dockerfile is ready to run with the docker build command. You should run the Docker container exposing it on port 8090. You can send your image under the name image_files within the POST to port 8090.

# Libraries and Resources Used
Python 3.10.10
Torch 2.0.1+CU118
aioflask==0.4.0
azure-storage-blob
Flask==2.1.3
Pillow
Protobuf==3.20.0
Uvicorn[standard]
OpenCV-Python-Headless
FVCore
Cloudpickle
OmegaConf
Werkzeug==2.0

<--------------------------------------------------------------------------------------------------------------->

# Hacim Hesaplama

Bu projede daha önceden eğitilmiş olan modelinizi kullanarak, Mask-RCNN ile tanımladıgınız nesnenin hacmini gerçek boyutlara en yakın değerleri alabilecek şekilde hesaplayabilirsiniz.

# Jupyter-Notebook

Projeyi notebook üzerinde çalıştırabilir ve predict.ipynb dosyası ile adımları takip ederek sonuçları gözlemleyebilirsiniz.

# Docker API Uygulaması

Ayrıca bu uygulamayı dockerize edebilirsiniz.
Dockerfile dosyası docker build komutu ile çalışmaya hazırdır.
Docker konteynerını 8090 portu ile expose ederek çalıştırmalısınız.
8090 portuna atacağınız POST içerisinde image_files adı altında görselinizi gönderebilirsiniz.

# Kullanılan Kütüphane ve Kaynaklar
Pyton 3.10.10
Torch2.0.1+CU118
aioflask==0.4.0
azure-storage-blob
Flask==2.1.3
pillow
protobuf==3.20.0
uvicorn[standard]
opencv-python-headless
fvcore
cloudpickle
omegaconf
Werkzeug==2.0

