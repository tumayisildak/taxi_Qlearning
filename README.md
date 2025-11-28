# 🚖 Custom Taxi Environment - Q-Learning Projesi

Pekiştirmeli öğrenme (Reinforcement Learning) ile otonom taksi simülasyonu. Taksi, Q-Learning algoritması kullanarak yolcuları herhangi bir hücreden alıp herhangi bir hücreye bırakmayı öğrenir.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Pygame](https://img.shields.io/badge/pygame-2.0+-red.svg)

## 🎯 Özellikler

- ✅ **Her hücreden yolcu al/bırak** - Tüm grid kullanılabilir
- ✅ **6×6 grid ortamı** - ULTRA BASİT (2 engel, 3 duvar)  
- ✅ **Grid %90 açık** - Taksi her yöne rahatça hareket eder
- ✅ **Q-Learning ile eğitim** - 100K episode
- ✅ **Gelişmiş görselleştirme** - Pygame ile gerçek zamanlı render
- ✅ **Taksi pozisyon koruma** - Yolcu bıraktıktan sonra aynı yerde kalır ⭐
- ✅ **Sürekli yolcu üretimi** - Sonsuz görev modu

## 📦 Kurulum

```bash
pip install numpy gym pygame
```

## 🚀 Hızlı Başlangıç (3 Adım)

### 1️⃣ Test Et
\`\`\`bash
python test_env.py
\`\`\`

### 2️⃣ Eğit (20-30 dakika)
\`\`\`bash
python train_qtable.py
\`\`\`

### 3️⃣ Çalıştır
\`\`\`bash
python run_taxi.py
\`\`\`

## 🎮 Ortam Detayları

### Grid Yapısı (v2.2 - ULTRA BASİT)
- **Boyut**: 6×6 (36 hücre)
- **Engeller**: Sadece 2 (kırmızı)
- **Duvarlar**: Sadece 3 küçük duvar
- **Açıklık**: %90 - Neredeyse tamamen açık!

### Aksiyonlar
- 0: Aşağı, 1: Yukarı, 2: Sağa, 3: Sola
- 4: Pickup, 5: Dropoff

### Ödüller (v2.2 - Güçlendirilmiş)
| Durum | Ödül |
|-------|------|
| Her adım | -0.5 |
| Geçersiz hareket | -15 |
| Başarılı pickup | +50 |
| Başarılı dropoff | +200 |

## 🧠 Q-Learning

### Hiperparametreler
- Episodes: 100,000
- Alpha: 0.15
- Gamma: 0.98
- Epsilon: 1.0 → 0.01

### Beklenen Performans
- **Başarı**: %95-97
- **Adım**: 10-18
- **Ödül**: +80...+120

## 🎨 Görselleştirme

- **Sarı Kare**: Taksi 🚖
- **Mavi (P)**: Yolcu 👤
- **Yeşil (D)**: Hedef 🎯
- **Kırmızı**: Engel ⛔
- **Gri Çizgi**: Duvar 🧱

## 💡 Önemli Özellik: Taksi Pozisyon Koruma

Yolcu bıraktıktan sonra taksi **aynı yerde kalır**, sadece yeni yolcu üretilir:

\`\`\`
🚖 Görev #1 başladı
   ✓ Görev tamamlandı!
   ⟳ Yeni yolcu üretiliyor (taksi aynı yerde)...

🚖 Görev #2 başladı
   (Taksi önceki konumda devam eder)
\`\`\`

## 🔧 Özelleştirme

### Ortam
\`\`\`python
# custom_taxi_env.py
self.blocked = {(2, 2), (3, 3)}  # Engeller
\`\`\`

### Eğitim
\`\`\`python
# train_qtable.py
episodes=150000  # Daha fazla eğitim
alpha=0.2        # Daha hızlı öğrenme
\`\`\`

### Ödüller
\`\`\`python
# custom_taxi_env.py - step() içinde
reward = 80   # pickup
reward = 300  # dropoff
\`\`\`

## 🐛 Sorun Giderme

### Pygame açılmıyor
\`\`\`bash
pip install --upgrade pygame
\`\`\`

### Q-table bulunamadı
\`\`\`bash
python train_qtable.py
\`\`\`

### Düşük başarı
1. Daha fazla episode (150K)
2. Ödülleri artır
3. Engelleri azalt

## 📁 Dosyalar

- \`custom_taxi_env.py\` - Ana ortam
- \`train_qtable.py\` - Eğitim
- \`run_taxi.py\` - Çalıştırma
- \`test_env.py\` - Test
- \`q_table.npy\` - Eğitilmiş model

## 📊 Grid Haritası

\`\`\`
  0   1   2   3   4   5
┌───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │ 0
├───┼───┼───┼───┼───┼───┤
│   │ ══│══ │   │   │   │ 1
├───┼───┼───┼───┼───┼───┤
│   │   │ X │   │   │   │ 2
├───┼───┼───┼───┼───┼───┤
│   │   │   │ X │   ║   │ 3
├───┼───┼───┼───┼───┼───┤
│   │   │   │ ══│══ │   │ 4
├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │ 5
└───┴───┴───┴───┴───┴───┘

X = Engel (2)
═ = Duvar (3)
Grid %90 açık!
\`\`\`

## 🎓 Öğrenme Kaynakları

- [Sutton & Barto - RL Kitabı](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## 📋 Hızlı Referans

\`\`\`bash
python test_env.py       # Test
python train_qtable.py   # Eğit (20-30 dk)
python run_taxi.py       # Çalıştır
rm q_table*.npy          # Temizle
\`\`\`

## 📜 Lisans

MIT License - Eğitim amaçlı, özgürce kullanılabilir.

---

**⭐ Beğendiyseniz yıldız verin!**

**Made with ❤️ and 🤖**
