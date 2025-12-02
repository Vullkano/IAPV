# -*- coding: utf-8 -*-
import os
import json
import time
import random
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
from groq import Groq
from dotenv import load_dotenv
import pyttsx3
import winsound

# --- 1. CONFIGURAÇÃO E INIT ---

# Carregar API Key - https://console.groq.com/keys ## TODO: Colocar a tua chave num ficheiro .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# Inicializar TTS (Voz)
tts_engine = None
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 140)
    tts_engine.setProperty('volume', 0.9)
    
    # Tentar encontrar uma voz feminina/suave
    voices = tts_engine.getProperty('voices')
    elira_voice_id = None
    preferred_names = ["zira", "samantha", "helena", "ava", "tessa", "espeak-english"]
    for voice in voices:
        if any(p in voice.name.lower() for p in preferred_names):
            elira_voice_id = voice.id
            break
    if elira_voice_id:
        tts_engine.setProperty('voice', elira_voice_id)
except Exception as e:
    print(f"Aviso: TTS não iniciou ({e}).")

def speak_text(text):
    """Tenta falar. Se o motor estiver ocupado, ignora para não travar o jogo."""
    if not tts_engine: return
    
    def speak_thread():
        try:
            # Em vez de tentar parar à força (o que encrava), apenas fala se possível
            tts_engine.say(text)
            tts_engine.runAndWait()
        except RuntimeError:
            # Se o motor já estiver a falar (loop running), ignoramos esta frase
            # Isto impede o erro "run loop already started" que congela tudo
            pass
        except Exception as e:
            print(f"Erro de voz: {e}")

    threading.Thread(target=speak_thread, daemon=True).start()

# --- 2. DADOS DO JOGO ---

WIDTH, HEIGHT = 18, 12  # TODO: Ajustar tamanho do mapa
CELL_SIZE = 50          # Reduzi um pouco para caber no ecrã
ASSET_DIR = "assets_50"
MEMORY_FILE = "elira_memory_rpg.json"

# --- 3. GESTÃO DE MEMÓRIA ---

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"conversations": [], "game_state": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

memory = load_memory()

# --- 4. CLASSE PRINCIPAL DO JOGO ---

class GameGUI:
    def __init__(self, master):
        self.root = master
        self.root.title("Elira: The Dark Library")
        self.root.configure(bg="#1a1a1a")

        # --- ESTADO DO JOGO (RPG) ---
        self.player_pos = [0, 0]
        self.hp = 100
        self.inventory = []
        self.game_over = False
        self.triggered_traps = set()

        # --- NOVAS VARIAVEIS DE MAGIA ---
        self.sight_range = 2          # O alcance agora é variável
        self.magic_traps = False      # Ver armadilhas
        self.magic_treasures = False  # Ver tesouros ao longe
        
        # Gerar o Nível
        self.walls, self.traps, self.treasures, self.exit_pos = self.generate_level()
        
        # Carregar Imagens
        self.load_assets()

        # --- INTERFACE (UI) - LAYOUT DE 2 COLUNAS ---
        
        # Container Principal
        main_container = tk.Frame(master, bg="#1a1a1a")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # COLUNA DA ESQUERDA (MAPA)
        left_frame = tk.Frame(main_container, bg="#1a1a1a")
        left_frame.pack(side=tk.LEFT, padx=10)

        # No __init__:
        self.canvas = tk.Canvas(left_frame, width=WIDTH * CELL_SIZE, height=HEIGHT * CELL_SIZE, 
                        bg="#202020", highlightthickness=0)
        self.canvas.pack()

        # COLUNA DA DIREITA (NPC + CHAT + STATUS)
        right_frame = tk.Frame(main_container, bg="#1a1a1a")
        right_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # 1. Imagem da Elira (Agora é um Label fixo, fora do mapa)
        # Nota: Certifica-te que a imagem 'idle' existe e tem bom tamanho (ex: 100x100 ou 150x150)
        self.npc_label = tk.Label(right_frame, image=self.imgs["idle"], bg="#1a1a1a")
        self.npc_label.pack(pady=5)

        # 2. Labels de Status
        self.status_var = tk.StringVar(value="HP: 100% | Inventário: Vazio")
        self.status_label = tk.Label(right_frame, textvariable=self.status_var, font=("Consolas", 10, "bold"), bg="#1a1a1a", fg="#00ff00")
        self.status_label.pack()
        
        self.objective_label = tk.Label(right_frame, text="OBJETIVO: Encontrar tesouros!", font=("Arial", 9, "bold"), bg="#1a1a1a", fg="#00ffff")
        self.objective_label.pack(pady=2)

        # 3. Chat Log
        self.chat_log = scrolledtext.ScrolledText(right_frame, height=15, width=40, state='disabled', 
                                                  bg="#2b2b2b", fg="#e0e0e0", font=("Verdana", 9), insertbackground="white")
        self.chat_log.pack(pady=5)
        
        # Tags de cor (Copia as mesmas de antes)
        self.chat_log.tag_config("player", foreground="#5dade2")
        self.chat_log.tag_config("elira", foreground="#f4d03f")
        self.chat_log.tag_config("danger", foreground="#e74c3c")
        self.chat_log.tag_config("success", foreground="#2ecc71")
        self.chat_log.tag_config("system", foreground="#95a5a6")

        # 4. Inputs e Botões
        self.entry_frame = tk.Frame(right_frame, bg="#1a1a1a")
        self.entry_frame.pack(pady=5)
        
        self.chat_entry = tk.Entry(self.entry_frame, width=25, bg="#404040", fg="white", insertbackground="white")
        self.chat_entry.pack(side=tk.LEFT, padx=5)
        self.chat_entry.bind("<Return>", self.talk_to_npc)

        # Botão Restart
        self.restart_btn = tk.Button(self.entry_frame, text="↻", command=self.reset_game, bg="#c0392b", fg="white", width=3)
        self.restart_btn.pack(side=tk.LEFT, padx=2)

        # NOVO: Botão Grimório (Ajuda)
        self.help_btn = tk.Button(self.entry_frame, text="?", command=self.show_grimoire, bg="#8e44ad", fg="white", width=3)
        self.help_btn.pack(side=tk.LEFT, padx=2)

        # --- 5. CONTROLO (Isto estava em falta!) ---
        master.bind("<Up>", lambda e: self.move_player("w"))
        master.bind("<Down>", lambda e: self.move_player("s"))
        master.bind("<Left>", lambda e: self.move_player("a"))
        master.bind("<Right>", lambda e: self.move_player("d"))

        # --- 6. ARRANQUE (Isto também faltava!) ---
        # Thread de animação da Elira
        threading.Thread(target=self.idle_animation, daemon=True).start()

        # Desenhar o mapa pela primeira vez (para não começar preto)
        self.draw_grid()
        self.log_message("System: Welcome to the Dark Library.", "system")

    def play_magic_sound(self, spell_type):
        """Gera sons sintéticos para os feitiços."""
        def sound_thread():
            if spell_type == "revelio":
                # Som agudo e rápido (Revelação)
                winsound.Beep(1000, 100)
                winsound.Beep(1500, 100)
                winsound.Beep(2000, 200)
            elif spell_type == "aurum":
                # Som de moeda (Ouro)
                winsound.Beep(2500, 100)
                winsound.Beep(4000, 300)
            elif spell_type == "lumen":
                # Som grave para agudo (Luz a expandir)
                for freq in range(400, 1000, 100):
                    winsound.Beep(freq, 40)
        
        threading.Thread(target=sound_thread, daemon=True).start()

    def generate_level(self):
        """Gera mapa e escolhe uma SAÍDA aleatória longe do jogador."""
        walls, traps, treasures = set(), set(), set()
        
        # 1. Gera o recheio (paredes/traps)
        empty_spots = [] # Lista para guardar sítios livres
        
        for r in range(HEIGHT):
            for c in range(WIDTH):
                if (r, c) == (0, 0): continue # Pula o jogador
                
                chance = random.random()
                if chance < 0.2: walls.add((r, c))
                elif chance < 0.25: traps.add((r, c))
                elif chance > 0.97: treasures.add((r, c))
                else:
                    empty_spots.append((r, c)) # Guardar chão vazio

        # 2. Define a Saída num sitio vazio aleatório (e remove parede se houver azar)
        if empty_spots:
            self.exit_pos = random.choice(empty_spots)
        else:
            self.exit_pos = (HEIGHT-1, WIDTH-1) # Fallback

        # Garante que não há parede/trap/tesouro em cima da saída
        if self.exit_pos in walls: walls.remove(self.exit_pos)
        if self.exit_pos in traps: traps.remove(self.exit_pos)
        if self.exit_pos in treasures: treasures.remove(self.exit_pos)

        # Garante pelo menos 1 tesouro
        if not treasures: treasures.add((HEIGHT//2, WIDTH//2))
        
        return walls, traps, treasures, self.exit_pos

    def load_assets(self):
        """Tenta carregar imagens. Se der erro, cria quadrados coloridos."""
        def load_img(name, color="magenta"):
            try:
                path = os.path.join(ASSET_DIR, name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} não existe")
                
                # Carrega e converte
                pil_img = Image.open(path).convert("RGBA").resize((CELL_SIZE, CELL_SIZE))
                return ImageTk.PhotoImage(pil_img)
            except Exception as e:
                print(f"Asset Error ({name}): {e}. Usando quadrado {color}.")
                # Fallback: Quadrado Sólido
                fallback = Image.new('RGB', (CELL_SIZE, CELL_SIZE), color=color)
                return ImageTk.PhotoImage(fallback)
        
        self.imgs = {
            "player": load_img("player.png", "cyan"),
            "wall": load_img("wall.png", "gray"),
            "treasure": load_img("treasure.png", "gold"),
            "idle": load_img("npc_idle.png", "pink"),
            "happy": load_img("npc_happy.png", "pink"),
            "anxious": load_img("npc_anxious.png", "pink"),
            "thinking": load_img("npc_thinking.png", "pink"),
            "encouraging": load_img("npc_encouraging.png", "pink"),
            "trap": load_img("trap.png", "red"),
        }

    def log_message(self, text, tag=None):
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, text + "\n", tag)
        self.chat_log.see(tk.END)
        self.chat_log.config(state='disabled')

    def draw_grid(self):
        self.canvas.delete("tile")
        
        # Garante alcance mínimo de 2 se a variável falhar
        current_sight = getattr(self, "sight_range", 2)

        for r in range(HEIGHT):
            for c in range(WIDTH):
                x, y = c * CELL_SIZE, r * CELL_SIZE
                dist = abs(self.player_pos[0] - r) + abs(self.player_pos[1] - c)
                
                # --- LÓGICA DE VISIBILIDADE ---
                is_visible = dist <= current_sight
                
                # 1. ZONA ESCURA (Fog of War)
                if not is_visible:
                    # Se tiver magia "Aurum", vê tesouros ao longe a amarelo
                    if getattr(self, "magic_treasures", False) and (r, c) in self.treasures:
                        self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="#f39c12", outline="black", tags="tile")
                    else:
                        # Desenha Preto (Escuridão) e salta para o próximo bloco
                        self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="black", outline="black", tags="tile")
                    continue 

                # 2. CHÃO (Se chegou aqui, é visível)
                # Desenha um quadrado azul escuro (Base)
                self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="#2c3e50", outline="#34495e", tags="tile")

                # 3. OBJETOS (Desenha Cor de Fundo + Imagem)
                
                # Portal de Saída
                if (r, c) == self.exit_pos:
                     self.canvas.create_rectangle(x+5, y+5, x+CELL_SIZE-5, y+CELL_SIZE-5, fill="#8e44ad", outline="white", width=2, tags="tile")

                # Paredes
                elif (r, c) in self.walls:
                    # Fundo Cinzento + Imagem
                    self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill="#7f8c8d", tags="tile")
                    if self.imgs.get("wall"):
                        self.canvas.create_image(x + CELL_SIZE // 2, y + CELL_SIZE // 2, image=self.imgs["wall"], tags="tile")
                
                # Armadilhas (Só se reveladas)
                elif (r, c) in self.traps:
                    show_trap = (r,c) in self.triggered_traps or getattr(self, "magic_traps", False)
                    if show_trap:
                        if self.imgs.get("trap"):
                            self.canvas.create_image(x + CELL_SIZE // 2, y + CELL_SIZE // 2, image=self.imgs["trap"], tags="tile")
                
                # Tesouros
                elif (r, c) in self.treasures:
                    self.canvas.create_oval(x+10, y+10, x+CELL_SIZE-10, y+CELL_SIZE-10, fill="#f1c40f", tags="tile")
                    if self.imgs.get("treasure"):
                        self.canvas.create_image(x + CELL_SIZE // 2, y + CELL_SIZE // 2, image=self.imgs["treasure"], tags="tile")

        # 4. JOGADOR
        px, py = self.player_pos[1] * CELL_SIZE, self.player_pos[0] * CELL_SIZE
        if self.imgs.get("player"):
            self.canvas.create_image(px + CELL_SIZE // 2, py + CELL_SIZE // 2, image=self.imgs["player"], tags="tile")

    
    def update_status(self):
        inv_text = f"Items: {len(self.inventory)}"
        color = "#00ff00" if self.hp > 50 else "#ff0000"
        self.status_var.set(f"HP: {self.hp}% | {inv_text}")
        self.status_label.config(fg=color)
        if self.treasures:
            self.objective_label.config(text=f"OBJETIVO: Faltam {len(self.treasures)} tesouros!", fg="#00ffff")
        else:
            self.objective_label.config(text="OBJETIVO: FOGE PELO PORTAL (AMARELO)!", fg="yellow")

    def move_player(self, cmd):
        if self.game_over: return

        r, c = self.player_pos
        new_r, new_c = r, c
        if cmd == "w": new_r -= 1
        elif cmd == "s": new_r += 1
        elif cmd == "a": new_c -= 1
        elif cmd == "d": new_c += 1

        moved = False
        if 0 <= new_r < HEIGHT and 0 <= new_c < WIDTH and (new_r, new_c) not in self.walls:
            self.player_pos[:] = [new_r, new_c]
            moved = True

            curr_pos = tuple(self.player_pos)

            if curr_pos in self.traps:
                self.triggered_traps.add(curr_pos)
                damage = 25
                self.hp -= damage
                self.log_message(f"💥 TRAP! -{damage} HP!", "danger")
                speak_text("Trap! Watch out!")
                self.player_pos[:] = [0, 0] 
                
                # ... dentro da lógica da armadilha ...
                if self.hp <= 0:
                    self.hp = 0
                    self.game_over = True
                    self.log_message("💀 GAME OVER.", "danger")
                    speak_text("No... please... don't die.")
                    
                    # ADICIONA ESTA LINHA:
                    self.root.after(1000, lambda: self.show_end_screen(victory=False))

            elif curr_pos in self.treasures:
                self.treasures.remove(curr_pos)
                item_name = random.choice(["Ancient Scroll", "Golden Chalice", "Mana Crystal"])
                self.inventory.append(item_name)
                self.log_message(f"✨ Found: {item_name}!", "success")
                
                if not self.treasures:
                    speak_text("That's the last one! Quick, to the portal!")
                else:
                    speak_text("A treasure! Need more though.")

            # --- NOVA LÓGICA DE SAÍDA ---
            elif curr_pos == self.exit_pos:
                if not self.treasures:
                    self.log_message("🌌 PORTAL OPENED! YOU ESCAPED!", "success")
                    speak_text("We are free! The light... it's beautiful!")
                    self.root.after(1000, lambda: self.show_end_screen(victory=True))
                    self.game_over = True
                else:
                    self.log_message("🔒 Portal Locked. Find all treasures first.", "system")
                    speak_text("The portal is sealed. We need all the artifacts to open it.")

        self.update_status()
        self.draw_grid()

        if moved and not self.game_over:
            self.spontaneous_npc_reaction()

    def get_proximity_status(self):
        pr, pc = self.player_pos
        status = []
        
        # 1. Onde ir (Tesouro ou Saída?)
        target = None
        target_name = ""
        
        if self.treasures:
            # Procura tesouro mais perto
            nearest_dist = float('inf')
            for tr, tc in self.treasures:
                d = abs(tr - pr) + abs(tc - pc)
                if d < nearest_dist:
                    nearest_dist = d
                    target = (tr, tc)
            target_name = "Treasure"
        else:
            # Se não há tesouros, o alvo é a SAÍDA
            target = self.exit_pos
            target_name = "EXIT PORTAL"

        # Calcular direção para o alvo
        if target:
            tr, tc = target
            d_lat = tr - pr
            d_lon = tc - pc
            dirs = []
            if d_lat < 0: dirs.append("North")
            elif d_lat > 0: dirs.append("South")
            if d_lon < 0: dirs.append("West")
            elif d_lon > 0: dirs.append("East")
            
            steps = abs(d_lat) + abs(d_lon)
            if steps == 0: status.append(f"GOAL REACHED: We are at the {target_name}!")
            else: status.append(f"GUIDANCE: {target_name} is to the {'-'.join(dirs)} ({steps} steps).")

        # 2. Sentido de Perigo
        nearby_traps = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (pr + dr, pc + dc) in self.traps:
                nearby_traps += 1
        if nearby_traps > 0:
            status.append(f"WARNING: {nearby_traps} hidden trap(s) ADJACENT!")

        return " | ".join(status)

    def talk_to_npc(self, event=None):
        if self.game_over: return
        msg = self.chat_entry.get().strip()
        if not msg: return
        
        self.chat_entry.delete(0, tk.END)
        self.log_message(f"You: {msg}", "player")

        # --- SISTEMA DE MAGIAS ---
        msg_low = msg.lower()
        
        if "revelio" in msg_low: # Poder 1: Ver Traps
            self.magic_traps = True
            self.play_magic_sound("revelio")
            self.log_message("✨ SPELL CAST: Traps revealed!", "success")
            speak_text("Behold! The snares are revealed.")
            self.draw_grid()
            return # Não envia para a IA

        elif "aurum" in msg_low: # Poder 2: Ver Tesouro Amarelo
            self.magic_treasures = True
            self.play_magic_sound("aurum")
            self.log_message("✨ SPELL CAST: Gold glitters in the dark!", "success")
            speak_text("Can you see the gold shining?")
            self.draw_grid()
            return

        elif "lumen" in msg_low: # Poder 3: Aumentar Visão
            self.sight_range += 1
            self.play_magic_sound("lumen")
            self.log_message("✨ SPELL CAST: Light expands!", "success")
            speak_text("Let there be light.")
            self.draw_grid()
            return

        # Se não for magia, chama a thread:
        print(f"--- A processar mensagem: {msg} ---") # Debug para ver se o Enter funciona
        self.run_npc_thread(msg, False)

    def show_grimoire(self):
        """Mostra uma janela com a lista de feitiços."""
        info = """
        📖 ELIRA'S GRIMOIRE 📖
        
        Commands & Spells:
        ---------------------------
        ➤ 'Revelio'
           Reveals all hidden traps on the map (Red Skulls).
           
        ➤ 'Aurum'
           Highlights distant treasures in GOLD, allowing
           you to see them in the dark.
           
        ➤ 'Lumen'
           Permanently increases your light radius by 1 step.
           
        ➤ 'Restart' (Button)
           Resets the dungeon layout completely.
           
        Talk to Elira for hints!
        """
        messagebox.showinfo("Grimoire", info)

    def spontaneous_npc_reaction(self):
        self.run_npc_thread("", True)

    def reset_game(self):
        """Reinicia todo o estado do jogo."""
        # Reset de variáveis
        self.hp = 100
        self.inventory = []
        self.game_over = False
        self.triggered_traps = set()
        self.player_pos = [0, 0]

        self.sight_range = 2
        self.magic_traps = False
        self.magic_treasures = False
        
        # Gerar novo mapa (Isto muda as paredes e tesouros de sitio!_get_npc_reply)
        self.walls, self.traps, self.treasures, self.exit_pos = self.generate_level()
        
        # Limpar Chat
        self.chat_log.config(state='normal')
        self.chat_log.delete(1.0, tk.END)
        self.chat_log.config(state='disabled')
        
        # Atualizar Ecrã
        self.log_message("System: --- GAME RESTARTED ---", "system")
        speak_text("Let's try again. Be careful this time.")
        self.draw_grid()
        self.update_status()

    # 1. Função que prepara e lança a thread (Não bloqueia o jogo)
    def run_npc_thread(self, player_msg, is_spontaneous):
        def task():
            game_status = self.get_proximity_status()
            hist_list = memory.get("conversations", [])
            recent_history = "\n".join(hist_list[-5:])
            
            # Ajuste para evitar falhas se a mensagem for vazia
            safe_msg = player_msg.replace('"', "'") # Troca aspas duplas por simples para não partir o JSON
            
            if is_spontaneous:
                context = "Player moved. React quickly."
                max_tok = 60
            else:
                context = f"CONVERSATION: Player said: '{safe_msg}'. Reply/Roleplay."
                max_tok = 150

            prompt = f"""
SYSTEM: You are Elira.
HP: {self.hp}%
STATUS: {game_status}
INSTRUCTIONS: Return VALID JSON ONLY. No markdown.
MEMORY: {recent_history}
CONTEXT: {context}
OUTPUT JSON: {{ "text": "...", "emotion": "idle/happy/anxious" }}
"""
            try:
                print(f"--- (Thread) A enviar pedido... ---") 
                
                # Chamada à API
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tok, 
                    temperature=0.8
                )
                
                content = response.choices[0].message.content.strip()
                print(f"--- (Thread) Recebido: {content[:50]}... ---") # Mostra só o inicio

                # Limpeza do JSON (caso a IA ponha ```json no inicio)
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "")
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Se falhar o JSON, usa o texto direto
                    print("--- JSON falhou, a usar texto direto ---")
                    data = {"text": content, "emotion": "idle"}

                self.root.after(0, lambda: self.finalize_npc_reply(player_msg, data, is_spontaneous))

            except Exception as e:
                print(f"❌ ERRO API: {e}")
                err_data = {"text": "A minha mente está nevoada...", "emotion": "anxious"}
                self.root.after(0, lambda: self.finalize_npc_reply(player_msg, err_data, is_spontaneous))

        threading.Thread(target=task, daemon=True).start()

    # 2. Função que recebe a resposta e atualiza o ecrã (Rodando no Thread Principal)
    def finalize_npc_reply(self, player_msg, data, is_spontaneous):
        if not is_spontaneous:
            memory["conversations"].append(f"P: {player_msg} | E: {data['text']}")
        
        self.log_message(f"Elira: {data['text']}", "elira")
        self.animate_npc(data.get("emotion", "idle"))
        speak_text(data["text"])

    # Substitui o animate_npc antigo por este:
    def animate_npc(self, emotion):
        # Agora alteramos a imagem do Label lateral
        img = self.imgs.get(emotion, self.imgs["idle"])
        self.npc_label.configure(image=img) 
        self.root.after(2000, lambda: self.npc_label.configure(image=self.imgs["idle"]))

    def idle_animation(self):
        while True:
            time.sleep(random.randint(4, 8))
            if not self.game_over:
                self.animate_npc("idle")

    def show_end_screen(self, victory):
        """Esconde o jogo e mostra uma janela de Fim dedicada."""
        self.game_over = True
        
        # 1. ESCONDER A JANELA PRINCIPAL (O Efeito que pediste)
        self.root.withdraw() 
        
        # Cores e Textos
        if victory:
            title = "🏆 VITÓRIA!"
            msg = "Parabéns!\nEscapaste da Biblioteca Escura com todos os tesouros."
            bg_color = "#f1c40f" # Dourado
            fg_color = "#1a1a1a" # Preto
        else:
            title = "💀 GAME OVER"
            msg = "A tua luz apagou-se...\nA escuridão consumiu-te."
            bg_color = "#922b21" # Vermelho Sangue
            fg_color = "white"

        # Criar a Janela Nova
        end_win = tk.Toplevel(self.root)
        end_win.title(title)
        end_win.geometry("400x300")
        end_win.configure(bg=bg_color)
        
        # Importante: Se fecharem esta janela no X, fecha o programa todo
        def close_all():
            self.root.destroy()
        end_win.protocol("WM_DELETE_WINDOW", close_all)

        # Conteúdo Visual
        tk.Label(end_win, text=title, font=("Arial", 24, "bold"), bg=bg_color, fg=fg_color).pack(pady=20)
        tk.Label(end_win, text=msg, font=("Verdana", 12), bg=bg_color, fg=fg_color, justify="center").pack(pady=10)

        # Botões
        btn_frame = tk.Frame(end_win, bg=bg_color)
        btn_frame.pack(pady=30)

        # Lógica de Reiniciar
        def restart_action():
            self.root.deiconify() # <--- TRAZ O JOGO DE VOLTA
            end_win.destroy()     # Fecha a janela de Game Over
            self.reset_game()     # Reinicia valores

        tk.Button(btn_frame, text="↻ Tentar de Novo", command=restart_action, 
                  font=("Arial", 12, "bold"), bg="white", fg="black", width=15).pack(pady=5)
        
        tk.Button(btn_frame, text="✖ Sair do Jogo", command=close_all, 
                  font=("Arial", 10), bg=bg_color, fg=fg_color, relief="flat").pack(pady=5)

        # Som
        def play_sound():
            try:
                if victory:
                    winsound.Beep(600, 100)
                    winsound.Beep(800, 100)
                    winsound.Beep(1200, 300)
                else:
                    winsound.Beep(300, 200)
                    winsound.Beep(150, 400)
            except: pass
        threading.Thread(target=play_sound, daemon=True).start()
    
# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = GameGUI(root)
    root.mainloop()