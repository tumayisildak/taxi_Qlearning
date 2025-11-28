import numpy as np
from custom_taxi_env import CustomTaxiEnv
import os
from datetime import datetime

def train_qtable(episodes=50000, alpha=0.1, gamma=0.95, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, save_interval=5000):
    """
    Q-Learning ile taksi eğitimi
    
    Args:
        episodes: Toplam eğitim episode sayısı
        alpha: Öğrenme oranı (learning rate)
        gamma: İndirim faktörü (discount factor)
        epsilon_start: Başlangıç exploration oranı
        epsilon_end: Minimum exploration oranı
        epsilon_decay: Epsilon azalma oranı
        save_interval: Her kaç episode'da bir kayıt yapılacağı
    """
    
    env = CustomTaxiEnv()
    
    # Q-table'ı başlat
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Eğitim istatistikleri
    rewards_history = []
    steps_history = []
    success_history = []
    epsilon = epsilon_start
    
    print("=" * 60)
    print("TAKSI Q-LEARNING EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Alpha (öğrenme oranı): {alpha}")
    print(f"Gamma (indirim faktörü): {gamma}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
    print("=" * 60)
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit
            
            # Adım at
            next_state, reward, done, _, info = env.step(action)
            
            # Q-değerini güncelle (Q-Learning update rule)
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            
            # Bellman denklemi
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            Q[state, action] = new_value
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Sonsuz döngü kontrolü
            if steps > 500:
                done = True
        
        # İstatistikleri kaydet
        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_history.append(1 if total_reward > 0 else 0)
        
        # Epsilon'u azalt (exploration'dan exploitation'a geçiş)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # İlerleme raporu
        if episode % save_interval == 0:
            avg_reward = np.mean(rewards_history[-save_interval:])
            avg_steps = np.mean(steps_history[-save_interval:])
            success_rate = np.mean(success_history[-save_interval:]) * 100
            
            print(f"Episode {episode}/{episodes}")
            print(f"  Ortalama Ödül: {avg_reward:.2f}")
            print(f"  Ortalama Adım: {avg_steps:.1f}")
            print(f"  Başarı Oranı: {success_rate:.1f}%")
            print(f"  Epsilon: {epsilon:.4f}")
            print("-" * 60)
            
            # Ara kayıt
            np.save(f"q_table_checkpoint_{episode}.npy", Q)
    
    # Final istatistikleri
    print("\n" + "=" * 60)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    
    # Son 1000 episode istatistikleri
    final_window = min(1000, episodes)
    final_avg_reward = np.mean(rewards_history[-final_window:])
    final_avg_steps = np.mean(steps_history[-final_window:])
    final_success_rate = np.mean(success_history[-final_window:]) * 100
    
    print(f"Son {final_window} Episode Ortalamaları:")
    print(f"  Ödül: {final_avg_reward:.2f}")
    print(f"  Adım: {final_avg_steps:.1f}")
    print(f"  Başarı Oranı: {final_success_rate:.1f}%")
    print("=" * 60)
    
    # Q-table'ı kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"q_table_{timestamp}.npy"
    np.save(filename, Q)
    np.save("q_table.npy", Q)  # Son sürüm için
    
    print(f"\n✓ Q-table kaydedildi: {filename}")
    print(f"✓ Q-table kaydedildi: q_table.npy (latest)")
    
    # İstatistikleri de kaydet
    stats = {
        'rewards': rewards_history,
        'steps': steps_history,
        'success': success_history,
        'hyperparameters': {
            'episodes': episodes,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay
        }
    }
    np.save(f"training_stats_{timestamp}.npy", stats)
    print(f"✓ Eğitim istatistikleri kaydedildi: training_stats_{timestamp}.npy")
    
    env.close()
    return Q, stats


if __name__ == "__main__":
    # Eğitimi başlat - iyileştirilmiş parametreler
    Q, stats = train_qtable(
        episodes=100000,     # Daha fazla episode (önceden 50000)
        alpha=0.15,          # Daha hızlı öğrenme (önceden 0.1)
        gamma=0.98,          # Gelecek ödüllere daha fazla önem (önceden 0.95)
        epsilon_start=1.0,   # %100 exploration ile başla
        epsilon_end=0.01,    # %1 exploration'a kadar düş
        epsilon_decay=0.9995, # Daha yavaş azalma (önceden 0.995)
        save_interval=10000  # Her 10K'da rapor (önceden 5000)
    )
    
    print("\nEğitim tamamlandı! Şimdi 'python run_taxi.py' ile test edebilirsiniz.")
