# Araç Algılama ve Hız Tespiti Projesi

Bu proje, video veya kamera görüntülerinde araçları tespit ederek hızlarını ölçmeyi amaçlar. Bilgisayarla görü (Computer Vision) ve derin öğrenme teknikleri kullanılarak, araçların gerçek zamanlı olarak tespiti ve hızlarının hesaplanması sağlanır.

## Özellikler

- Yol üzerindeki araçları tespit eder.
- Araçların hızını tahmin eder ve raporlar.
- Video dosyası veya canlı kamera görüntüsü ile çalışabilir.
- YOLOv8 gibi modern nesne algılama modelleri kullanır.
- Sonuçları görselleştirir ve kaydeder.

## Kullanılan Teknolojiler

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Numpy

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
2. YOLOv8 ağırlık dosyasını (`yolov8n.pt`) proje dizinine ekleyin.

## Kullanım

1. Projeyi çalıştırmak için:
    ```bash
    python main.py
    ```
2. Video dosyasını veya kamera kaynağını belirtin.
3. Sonuçlar ekranda gösterilecek ve istenirse kaydedilecektir.

## Dosya Yapısı

- `main.py` : Ana uygulama dosyası
- `Speed_Detection/` : Araç tespiti ve hız hesaplama ile ilgili modüller
- `yolov8n.pt` : YOLOv8 ağırlık dosyası


