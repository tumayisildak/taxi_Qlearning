import numpy as np
import gym
from gym import Env, spaces
import pygame


class CustomTaxiEnv(Env):
    """
    Özel Taksi Ortamı - Gelişmiş Versiyon
    - Her hücreden yolcu alınabilir, her hücreye bırakılabilir
    - 6x6 grid
    - Duvarlar ve girilemez bölgeler
    - Görsel labirent tasarımı
    """
    
    def __init__(self, grid_size=6, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.rows = grid_size
        self.cols = grid_size

        # Girilemez hücreler (kırmızı) - MİNİMAL (sadece 2 engel)
        self.blocked = {
            (2, 2),
            (3, 3),
        }

        # Duvarlar (hücreler arası engeller)
        # Format: ((row1, col1), (row2, col2)) - bu iki hücre arasında duvar var
        self.walls = set()
        self._create_walls()

        # Actions: 0=down, 1=up, 2=right, 3=left, 4=pickup, 5=dropoff
        self.action_space = spaces.Discrete(6)

        # State space: taxi_row * taxi_col * passenger_row * passenger_col * 
        #              passenger_in_taxi * dest_row * dest_col
        # Ancak pratik olarak encode/decode kullanacağız
        max_states = (self.rows * self.cols * 
                     self.rows * self.cols * 2 * 
                     self.rows * self.cols)
        self.observation_space = spaces.Discrete(max_states)

        self.render_mode = render_mode

        # Pygame init
        pygame.init()
        self.cell_size = 100
        self.wall_thickness = 8
        self.window_size = (self.cols * self.cell_size,
                            self.rows * self.cell_size)
        self.window = None
        self.clock = pygame.time.Clock()
        self.font = None

        self.reset()

    def _create_walls(self):
        """Labirent duvarlarını oluştur - ÇOK MİNİMAL versiyon (az duvar)"""
        # Sadece birkaç stratejik duvar - navigasyon çok kolay
        walls_horizontal = [
            ((1, 1), (1, 2)),  # Küçük bir engel
            ((4, 3), (4, 4)),  # Küçük bir engel
        ]
        
        walls_vertical = [
            ((2, 4), (3, 4)),  # Küçük bir engel
        ]
        
        for wall in walls_horizontal + walls_vertical:
            self.walls.add(wall)
            # Duvarı ters yönde de ekle
            self.walls.add((wall[1], wall[0]))

    def _is_valid_position(self, row, col):
        """Pozisyon geçerli mi kontrol et"""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        if (row, col) in self.blocked:
            return False
        return True

    def _has_wall_between(self, pos1, pos2):
        """İki pozisyon arasında duvar var mı?"""
        return (pos1, pos2) in self.walls

    def encode(self, taxi_row, taxi_col, pass_row, pass_col, in_taxi, dest_row, dest_col):
        """State'i tek bir integer'a encode et"""
        i = taxi_row
        i = i * self.cols + taxi_col
        i = i * self.rows + pass_row
        i = i * self.cols + pass_col
        i = i * 2 + (1 if in_taxi else 0)
        i = i * self.rows + dest_row
        i = i * self.cols + dest_col
        return i

    def decode(self, state):
        """Integer state'i bileşenlerine ayır"""
        dest_col = state % self.cols
        state //= self.cols
        dest_row = state % self.rows
        state //= self.rows
        in_taxi = bool(state % 2)
        state //= 2
        pass_col = state % self.cols
        state //= self.cols
        pass_row = state % self.rows
        state //= self.rows
        taxi_col = state % self.cols
        state //= self.cols
        taxi_row = state
        return taxi_row, taxi_col, pass_row, pass_col, in_taxi, dest_row, dest_col

    def reset(self, seed=None, options=None):
        """Ortamı sıfırla"""
        super().reset(seed=seed)

        # Taxi spawn - geçerli bir pozisyonda
        valid_positions = [(r, c) for r in range(self.rows) 
                          for c in range(self.cols) 
                          if self._is_valid_position(r, c)]
        
        taxi_pos = valid_positions[np.random.randint(len(valid_positions))]
        self.taxi_row, self.taxi_col = taxi_pos

        # Passenger spawn - geçerli ve taxi'den farklı
        while True:
            pass_pos = valid_positions[np.random.randint(len(valid_positions))]
            if pass_pos != taxi_pos:
                self.pass_row, self.pass_col = pass_pos
                break

        # Destination - geçerli ve passenger'dan farklı
        while True:
            dest_pos = valid_positions[np.random.randint(len(valid_positions))]
            if dest_pos != (self.pass_row, self.pass_col):
                self.dest_row, self.dest_col = dest_pos
                break

        self.passenger_in_taxi = False
        self.terminated = False
        self.total_reward = 0
        self.step_count = 0

        return self._get_state(), {}

    def reset_passenger(self):
        """
        Sadece yolcu ve hedefi yenile, taksiyi hareket ettirme.
        Taksi bıraktığı yerde kalır, yeni yolcu üretilir.
        """
        # Geçerli pozisyonlar
        valid_positions = [(r, c) for r in range(self.rows) 
                          for c in range(self.cols) 
                          if self._is_valid_position(r, c)]
        
        # Yeni yolcu pozisyonu - taksi pozisyonundan farklı olmalı
        while True:
            pass_pos = valid_positions[np.random.randint(len(valid_positions))]
            if pass_pos != (self.taxi_row, self.taxi_col):
                self.pass_row, self.pass_col = pass_pos
                break

        # Yeni hedef - yolcu pozisyonundan farklı olmalı
        while True:
            dest_pos = valid_positions[np.random.randint(len(valid_positions))]
            if dest_pos != (self.pass_row, self.pass_col):
                self.dest_row, self.dest_col = dest_pos
                break

        self.passenger_in_taxi = False
        self.terminated = False
        self.total_reward = 0
        self.step_count = 0

        return self._get_state(), {}

    def _get_state(self):
        """Mevcut durumu encode et"""
        return self.encode(
            self.taxi_row, self.taxi_col,
            self.pass_row, self.pass_col,
            self.passenger_in_taxi,
            self.dest_row, self.dest_col
        )

    def step(self, action):
        """Bir adım at"""
        self.step_count += 1
        reward = -0.5  # Daha küçük zaman cezası (önceden -1)
        done = False

        old_pos = (self.taxi_row, self.taxi_col)
        new_row, new_col = self.taxi_row, self.taxi_col

        # Hareket aksiyonları
        if action == 0:  # down
            new_row = self.taxi_row + 1
        elif action == 1:  # up
            new_row = self.taxi_row - 1
        elif action == 2:  # right
            new_col = self.taxi_col + 1
        elif action == 3:  # left
            new_col = self.taxi_col - 1

        # Hareket geçerliliğini kontrol et
        if action < 4:  # Hareket aksiyonlarıysa
            new_pos = (new_row, new_col)
            
            # Grid dışı veya girilemez bölge kontrolü
            if not self._is_valid_position(new_row, new_col):
                reward = -15  # Daha büyük ceza (önceden -10)
            # Duvar kontrolü
            elif self._has_wall_between(old_pos, new_pos):
                reward = -15  # Daha büyük ceza (önceden -10)
            else:
                # Geçerli hareket
                self.taxi_row, self.taxi_col = new_row, new_col

        # Pickup
        elif action == 4:
            if not self.passenger_in_taxi:
                if (self.taxi_row, self.taxi_col) == (self.pass_row, self.pass_col):
                    # Başarılı pickup
                    self.passenger_in_taxi = True
                    reward = 50  # ÇOK YÜKSEK ödül (önceden 30)
                else:
                    # Yanlış yerde pickup denemesi
                    reward = -10
            else:
                # Zaten yolcu var
                reward = -10

        # Dropoff
        elif action == 5:
            if self.passenger_in_taxi:
                if (self.taxi_row, self.taxi_col) == (self.dest_row, self.dest_col):
                    # Başarılı dropoff - görev tamamlandı!
                    reward = 200  # ÇOK BÜYÜK ödül (önceden 100)
                    done = True
                else:
                    # Yanlış yerde dropoff
                    reward = -10
            else:
                # Yolcu yok, dropoff yapılamaz
                reward = -10

        self.total_reward += reward

        # Çok uzun sürerse timeout
        if self.step_count > 200:
            done = True
            reward -= 10  # Daha az ceza (önceden -20)

        return self._get_state(), reward, done, False, {
            'step_count': self.step_count,
            'total_reward': self.total_reward
        }

    def render(self):
        """Ortamı görselleştir"""
        if self.window is None:
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Custom Taxi Environment - Advanced")
            self.font = pygame.font.Font(None, 24)

        # Renkler
        COLOR_BG = (240, 240, 240)
        COLOR_ROAD = (255, 255, 255)
        COLOR_BLOCKED = (200, 50, 50)  # Kırmızı - girilemez
        COLOR_WALL = (60, 60, 60)  # Koyu gri - duvarlar
        COLOR_GRID = (200, 200, 200)
        COLOR_TAXI = (255, 200, 0)  # Sarı
        COLOR_PASSENGER = (0, 120, 255)  # Mavi
        COLOR_DESTINATION = (0, 200, 0)  # Yeşil
        COLOR_TEXT = (0, 0, 0)

        self.window.fill(COLOR_BG)

        # Hücreleri çiz
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size
                y = r * self.cell_size

                # Hücre rengi
                if (r, c) in self.blocked:
                    color = COLOR_BLOCKED
                else:
                    color = COLOR_ROAD

                pygame.draw.rect(self.window, color,
                               (x, y, self.cell_size, self.cell_size))
                
                # Grid çizgileri (ince)
                pygame.draw.rect(self.window, COLOR_GRID,
                               (x, y, self.cell_size, self.cell_size), 1)

        # Duvarları çiz
        for wall in self.walls:
            pos1, pos2 = wall
            r1, c1 = pos1
            r2, c2 = pos2
            
            # Yatay duvar mı dikey duvar mı?
            if r1 == r2:  # Yatay duvar (hücreler yan yana)
                # Sağdaki hücre
                wall_x = max(c1, c2) * self.cell_size
                wall_y = r1 * self.cell_size
                pygame.draw.rect(self.window, COLOR_WALL,
                               (wall_x - self.wall_thickness//2, wall_y,
                                self.wall_thickness, self.cell_size))
            else:  # Dikey duvar (hücreler alt alta)
                # Alttaki hücre
                wall_x = c1 * self.cell_size
                wall_y = max(r1, r2) * self.cell_size
                pygame.draw.rect(self.window, COLOR_WALL,
                               (wall_x, wall_y - self.wall_thickness//2,
                                self.cell_size, self.wall_thickness))

        # Destination işareti (yeşil daire)
        dest_x = self.dest_col * self.cell_size + self.cell_size // 2
        dest_y = self.dest_row * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.window, COLOR_DESTINATION, (dest_x, dest_y), 18)
        pygame.draw.circle(self.window, COLOR_TEXT, (dest_x, dest_y), 18, 2)
        
        # "D" harfi
        text = self.font.render("D", True, COLOR_TEXT)
        text_rect = text.get_rect(center=(dest_x, dest_y))
        self.window.blit(text, text_rect)

        # Passenger (eğer taksiye binmemişse)
        if not self.passenger_in_taxi:
            pass_x = self.pass_col * self.cell_size + self.cell_size // 2
            pass_y = self.pass_row * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.window, COLOR_PASSENGER, (pass_x, pass_y), 18)
            pygame.draw.circle(self.window, COLOR_TEXT, (pass_x, pass_y), 18, 2)
            
            # "P" harfi
            text = self.font.render("P", True, COLOR_TEXT)
            text_rect = text.get_rect(center=(pass_x, pass_y))
            self.window.blit(text, text_rect)

        # Taxi (sarı kare)
        taxi_x = self.taxi_col * self.cell_size + self.cell_size // 4
        taxi_y = self.taxi_row * self.cell_size + self.cell_size // 4
        taxi_size = self.cell_size // 2
        
        pygame.draw.rect(self.window, COLOR_TAXI,
                        (taxi_x, taxi_y, taxi_size, taxi_size))
        pygame.draw.rect(self.window, COLOR_TEXT,
                        (taxi_x, taxi_y, taxi_size, taxi_size), 3)
        
        # Eğer yolcu taksideyse, taxi'nin üzerinde "P" göster
        if self.passenger_in_taxi:
            text = self.font.render("P", True, COLOR_PASSENGER)
            text_rect = text.get_rect(center=(taxi_x + taxi_size//2, taxi_y + taxi_size//2))
            self.window.blit(text, text_rect)

        # Bilgi metni
        info_text = f"Steps: {self.step_count} | Reward: {self.total_reward:.0f}"
        text_surface = self.font.render(info_text, True, COLOR_TEXT)
        self.window.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS

    def close(self):
        """Pygame penceresini kapat"""
        if self.window is not None:
            pygame.quit()
            self.window = None
