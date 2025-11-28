import numpy as np
from custom_taxi_env import CustomTaxiEnv
import pygame
import time

def test_environment():
    """
    Ortamı test et - rastgele aksiyonlarla
    """
    print("=" * 60)
    print("ORTAM TEST EDİLİYOR")
    print("=" * 60)
    
    env = CustomTaxiEnv()
    
    # Ortam bilgileri
    print(f"\nOrtam Bilgileri:")
    print(f"  Grid boyutu: {env.rows}x{env.cols}")
    print(f"  Girilemez hücre sayısı: {len(env.blocked)}")
    print(f"  Duvar sayısı: {len(env.walls) // 2}")  # Her duvar iki yönde
    print(f"  Action space: {env.action_space.n} aksiyon")
    print(f"  Observation space: {env.observation_space.n} state")
    
    print(f"\nGirilemez hücreler (kırmızı):")
    for pos in sorted(env.blocked):
        print(f"  {pos}")
    
    print(f"\nAksiyonlar:")
    print(f"  0: Aşağı")
    print(f"  1: Yukarı")
    print(f"  2: Sağa")
    print(f"  3: Sola")
    print(f"  4: Yolcu al (Pickup)")
    print(f"  5: Yolcu bırak (Dropoff)")
    
    print("\n" + "=" * 60)
    print("RASTGELE AKSİYONLAR TEST EDİLİYOR")
    print("Pencereyi kapatarak sonlandırabilirsiniz")
    print("=" * 60)
    
    # Test episodes
    num_test_episodes = 3
    
    for episode in range(1, num_test_episodes + 1):
        print(f"\n--- Test Episode {episode}/{num_test_episodes} ---")
        
        state, info = env.reset()
        env.render()
        
        print(f"Başlangıç durumu:")
        print(f"  Taksi: ({env.taxi_row}, {env.taxi_col})")
        print(f"  Yolcu: ({env.pass_row}, {env.pass_col})")
        print(f"  Hedef: ({env.dest_row}, {env.dest_col})")
        print(f"  State ID: {state}")
        
        done = False
        step = 0
        total_reward = 0
        
        time.sleep(1)
        
        # Rastgele aksiyonlar
        while not done and step < 50:  # Maksimum 50 adım
            # Event kontrolü
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            action = env.action_space.sample()
            
            action_names = ['Aşağı', 'Yukarı', 'Sağa', 'Sola', 'Pickup', 'Dropoff']
            print(f"\nAdım {step + 1}: {action_names[action]} (action={action})")
            
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Ödül: {reward:+.1f}")
            print(f"  Toplam ödül: {total_reward:+.1f}")
            print(f"  Taksi pozisyonu: ({env.taxi_row}, {env.taxi_col})")
            print(f"  Yolcu takside: {'Evet' if env.passenger_in_taxi else 'Hayır'}")
            
            if done:
                print(f"\n🎉 Episode tamamlandı!")
                print(f"  Toplam adım: {step + 1}")
                print(f"  Final ödül: {total_reward:+.1f}")
            
            env.render()
            time.sleep(0.5)
            
            state = next_state
            step += 1
        
        if not done:
            print(f"\n⏸ Episode maksimum adıma ulaştı ({step} adım)")
        
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)
    env.close()


def test_specific_scenario():
    """
    Belirli bir senaryoyu test et
    """
    print("\n" + "=" * 60)
    print("ÖZEL SENARYO TESTİ")
    print("=" * 60)
    
    env = CustomTaxiEnv()
    
    # Manuel state ayarla
    env.taxi_row, env.taxi_col = 0, 0
    env.pass_row, env.pass_col = 5, 5
    env.dest_row, env.dest_col = 0, 5
    env.passenger_in_taxi = False
    
    print(f"\nSenaryo:")
    print(f"  Taksi başlangıç: ({env.taxi_row}, {env.taxi_col})")
    print(f"  Yolcu konumu: ({env.pass_row}, {env.pass_col})")
    print(f"  Hedef konum: ({env.dest_row}, {env.dest_col})")
    
    env.render()
    time.sleep(2)
    
    # Test aksiyonları
    test_actions = [
        (0, "Aşağı git"),
        (2, "Sağa git"),
        (4, "Pickup dene (başarısız olmalı - yanlış konum)"),
    ]
    
    for action, description in test_actions:
        print(f"\n{description}")
        state, reward, done, _, info = env.step(action)
        print(f"  Ödül: {reward:+.1f}")
        print(f"  Pozisyon: ({env.taxi_row}, {env.taxi_col})")
        env.render()
        time.sleep(1)
    
    env.close()


if __name__ == "__main__":
    try:
        # Genel test
        test_environment()
        
        # Özel senaryo testi (opsiyonel)
        # test_specific_scenario()
        
    except KeyboardInterrupt:
        print("\n\n⏸ Test durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
