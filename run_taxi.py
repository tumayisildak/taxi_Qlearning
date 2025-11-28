import numpy as np
from custom_taxi_env import CustomTaxiEnv
import pygame
import time
import os

def run_trained_taxi(q_table_path="q_table.npy", delay=0.3, max_episodes=None):
    """
    Eğitilmiş Q-table ile taksiyi çalıştır
    
    Args:
        q_table_path: Q-table dosya yolu
        delay: Her adım arasındaki bekleme süresi (saniye)
        max_episodes: Maksimum görev sayısı (None = sonsuz)
    """
    
    # Q-table'ı yükle
    if not os.path.exists(q_table_path):
        print(f"HATA: {q_table_path} bulunamadı!")
        print("Önce 'python train_qtable.py' ile eğitim yapın.")
        return
    
    Q = np.load(q_table_path)
    print(f"✓ Q-table yüklendi: {q_table_path}")
    print(f"  Q-table boyutu: {Q.shape}")
    print(f"  Toplam öğrenilen state sayısı: {np.count_nonzero(Q)}")
    
    env = CustomTaxiEnv()
    
    print("\n" + "=" * 60)
    print("OTONOM TAKSİ ÇALIŞIYOR")
    print("=" * 60)
    print("Pencereyi kapatarak çıkabilirsiniz.")
    print("=" * 60 + "\n")
    
    episode = 0
    total_rewards = []
    total_steps = []
    
    try:
        # İlk görev
        state, _ = env.reset()
        env.render()
        
        while True:
            episode += 1
            done = False
            step_count = 0
            episode_reward = 0
            
            print(f"\n🚖 Görev #{episode} başladı")
            print(f"   Yolcu: ({env.pass_row}, {env.pass_col})")
            print(f"   Hedef: ({env.dest_row}, {env.dest_col})")
            
            # Tek görev döngüsü
            while not done:
                # Event kontrolü (pencere kapatma)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # En iyi aksiyonu seç (exploitation only, no exploration)
                action = np.argmax(Q[state])
                
                # Adım at
                state, reward, done, _, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Görselleştir
                env.render()
                time.sleep(delay)
                
                # Sonsuz döngü kontrolü
                if step_count > 30:
                    print("   ⚠️  Çok uzun sürdü, yeni görev başlatılıyor...")
                    break
            
            # Görev tamamlandı
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
            
            if done and episode_reward > 0:
                print(f"   ✓ Görev tamamlandı!")
            else:
                print(f"   ✗ Görev başarısız")
            
            print(f"   Adım sayısı: {step_count}")
            print(f"   Toplam ödül: {episode_reward:.1f}")
            
            # İstatistikler
            if episode % 5 == 0:
                avg_reward = np.mean(total_rewards[-5:])
                avg_steps = np.mean(total_steps[-5:])
                success_rate = sum(1 for r in total_rewards[-5:] if r > 0) / 5 * 100
                print(f"\n📊 Son 5 görev istatistikleri:")
                print(f"   Ortalama ödül: {avg_reward:.1f}")
                print(f"   Ortalama adım: {avg_steps:.1f}")
                print(f"   Başarı oranı: {success_rate:.0f}%")
            
            # Maksimum episode kontrolü
            if max_episodes and episode >= max_episodes:
                print(f"\n✓ {max_episodes} görev tamamlandı, program sonlandırılıyor.")
                break
            
            # Yeni yolcu üret (TAKSİ AYNI YERDE KALIR)
            print(f"   ⟳ Yeni yolcu üretiliyor (taksi aynı yerde)...")
            time.sleep(0.5)
            state, _ = env.reset_passenger()  # Sadece yolcu değişir, taksi kalmaz
            env.render()
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n⏸ Program durduruldu.")
    
    finally:
        # Final istatistikleri
        if total_rewards:
            print("\n" + "=" * 60)
            print("FINAL İSTATİSTİKLER")
            print("=" * 60)
            print(f"Toplam görev: {len(total_rewards)}")
            print(f"Ortalama ödül: {np.mean(total_rewards):.2f}")
            print(f"Ortalama adım: {np.mean(total_steps):.1f}")
            success_count = sum(1 for r in total_rewards if r > 0)
            print(f"Başarılı görev: {success_count}/{len(total_rewards)} ({success_count/len(total_rewards)*100:.1f}%)")
            print(f"En iyi ödül: {max(total_rewards):.1f}")
            print(f"En kötü ödül: {min(total_rewards):.1f}")
            print("=" * 60)
        
        env.close()
        print("\n✓ Program sonlandı.")


if __name__ == "__main__":
    # Farklı kullanım örnekleri:
    
    # 1. Normal kullanım (sonsuz)
    run_trained_taxi(delay=0.2)
    
    # 2. Belirli sayıda görev
    # run_trained_taxi(delay=0.2, max_episodes=10)
    
    # 3. Daha hızlı (kısa bekleme)
    # run_trained_taxi(delay=0.1)
    
    # 4. Daha yavaş (detaylı izleme)
    # run_trained_taxi(delay=0.5)
    
    # 5. Özel Q-table dosyası
    # run_trained_taxi(q_table_path="q_table_20241128_143000.npy", delay=0.2)
